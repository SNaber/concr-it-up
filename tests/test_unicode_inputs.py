"""Regression coverage for text inputs shared by the CLI and hosted GUI."""

import codecs
import csv
import json
from pathlib import Path

import pytest

from apps.core_gui.hosted_config import HostedLimits, inspect_gold, inspect_holdout_inputs, inspect_target
from concreteness_knn_core.config import load_config
from concreteness_knn_core.data import load_gold_df, load_vocab_words
from concreteness_knn_core.prediction import PredictionPipeline
from concreteness_knn_core.prediction.prediction_holdout_eval import PredictionHoldoutEvaluator


ENCODINGS = [
    ("utf-8", b""),
    ("utf-8", codecs.BOM_UTF8),
    ("utf-16-le", codecs.BOM_UTF16_LE),
    ("utf-16-be", codecs.BOM_UTF16_BE),
    ("utf-32-le", codecs.BOM_UTF32_LE),
    ("utf-32-be", codecs.BOM_UTF32_BE),
]
WORDS = ["Äpfel", "Straße", "café", "l’été", "O'Neill", "co-operate", "東京", "قلب", "می\u200cروم", "👩\u200d💻"]


def dataset(path, *, lowercase=False):
    return {"gold": str(path), "word_column": "word", "score_column": "score",
            "lowercase": lowercase, "pos_filter": {"enabled": False}}


@pytest.mark.parametrize("encoding,bom", ENCODINGS)
def test_unicode_files_match_between_admission_and_loaders(tmp_path, encoding, bom):
    target = tmp_path / "targets.txt"
    # A decomposed accent must match the precomposed form, without stripping it.
    target.write_bytes(bom + ("\r\n".join(WORDS + ["cafe\u0301"]) + "\r\n").encode(encoding))
    assert load_vocab_words(str(target), lowercase=False) == WORDS + ["café"]
    counts = inspect_target(target, lowercase=False, limits=HostedLimits())
    assert counts["normalized_unique_rows"] == len(WORDS)

    gold = tmp_path / "gold.tsv"
    rows = ["word\tscore"] + [f"{word}\t2" for word in WORDS] + ["cafe\u0301\t4"]
    gold.write_bytes(bom + "\n".join(rows).encode(encoding))
    cfg = dataset(gold)
    frame = load_gold_df(cfg).set_index("word")
    assert set(frame.index) == set(WORDS)
    assert frame.loc["café", "score"] == 3
    counts = inspect_gold({"dataset": cfg}, HostedLimits())
    assert counts["normalized_unique_rows"] == len(frame)
    assert counts["retained_rows"] == len(WORDS) + 1


@pytest.mark.parametrize("words", [["NA", "null", "N/A", "NaN"], ["001", "002"]])
def test_csv_words_are_literal_text(tmp_path, words):
    gold = tmp_path / "gold.csv"
    gold.write_text("word,score\n" + "\n".join(f"{word},2" for word in words), encoding="utf-8")
    assert set(load_gold_df(dataset(gold))["word"]) == set(words)


@pytest.mark.parametrize("payload,message", [(b"word,score\nApf\xe4l,2\n", "UTF-8"),
                                             (b"word,score\napple\x00suffix,2\n", "NUL")])
def test_invalid_text_is_rejected_without_silent_corruption(tmp_path, payload, message):
    path = tmp_path / "input.csv"
    path.write_bytes(payload)
    with pytest.raises(ValueError, match=message):
        load_gold_df(dataset(path))
    with pytest.raises(ValueError, match=message):
        load_vocab_words(str(path), lowercase=False)


@pytest.mark.parametrize("encoding,bom", ENCODINGS)
def test_config_accepts_unicode_encoding_markers(tmp_path, encoding, bom):
    config = tmp_path / "config.json"
    config.write_bytes(bom + json.dumps({"dataset": {"word_column": "Wörter"}}, ensure_ascii=False).encode(encoding))
    assert load_config(str(config))["dataset"]["word_column"] == "Wörter"


@pytest.mark.parametrize("lowercase", [False, True])
def test_normalization_preserves_distinct_words_and_case_setting(tmp_path, lowercase):
    target = tmp_path / "targets.txt"
    words = ["Ä", "A\u0308", "a", "ß", "ss", "’", "'", "－", "-", "①", "1"]
    target.write_text("\n".join(words), encoding="utf-8")
    expected = ["Ä", "Ä", *words[2:]]
    if lowercase:
        expected = [word.lower() for word in expected]
    assert load_vocab_words(str(target), lowercase) == expected


def test_empty_required_cells_are_skipped_consistently(tmp_path):
    gold = tmp_path / "gold.csv"
    gold.write_text('word,score\nNA,2\n,3\n"  ",4\nfoo,\nbar,  \n', encoding="utf-8")
    cfg = dataset(gold)
    assert list(load_gold_df(cfg)["word"]) == ["NA"]
    assert inspect_gold({"dataset": cfg}, HostedLimits())["retained_rows"] == 1


def test_holdout_uses_the_same_unicode_keys_as_admission(tmp_path):
    holdout = tmp_path / "holdout.csv"
    predictions = tmp_path / "predictions.csv"
    holdout.write_text("word,score\ncafe\u0301,1\nNA,2\nnull,3\n001,4\n", encoding="utf-16")
    predictions.write_text("word,pred\ncafé,1\nNA,2\nnull,3\n001,4\n", encoding="utf-8-sig")
    cfg = {"dataset": dataset(holdout), "reports": {"level": "full"},
           "prediction": {"holdout": str(holdout), "predictions_csv": str(predictions),
                          "prediction_column": "pred"}}
    assert inspect_holdout_inputs(cfg, HostedLimits())["n_matched"] == 4
    result = PredictionHoldoutEvaluator().evaluate_holdout(cfg, "test.json", "7.0.0")
    assert result.summary_payload["n_matched"] == 4
    assert result.summary_payload["rmse"] == 0
    assert set(result.merged_df["_word"]) == {"café", "NA", "null", "001"}

    holdout.write_text("word,score\ncafé,1\ncafe\u0301,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate words"):
        inspect_holdout_inputs(cfg, HostedLimits())
    with pytest.raises(ValueError, match="Duplicate words"):
        PredictionHoldoutEvaluator().evaluate_holdout(cfg, "test.json", "7.0.0")


@pytest.mark.parametrize("kind", ["vec", "ft_bin"])
@pytest.mark.parametrize("lowercase", [False, True])
@pytest.mark.parametrize("gold_encoding", ["utf-8-sig", "utf-16"])
def test_unicode_prediction_roundtrip(tmp_path, kind, lowercase, gold_encoding):
    """Exercise real vector lookup, training, and output CSV serialization."""
    example = Path(__file__).resolve().parents[1] / "examples" / "smoke"
    cfg = load_config(str(example / "config.json"))
    words = ["Äpfel", "Straße", "café", "l’été", "O'Neill", "東京", "قلب", "NA", "null", "001"]
    expected = [word.lower() for word in words] if lowercase else words
    vector_lines = (example / "mini.vec").read_text(encoding="utf-8").splitlines()
    vectors = tmp_path / "mini.vec"
    vectors.write_text(vector_lines[0] + "\n" + "\n".join(
        word + " " + line.split(" ", 1)[1] for word, line in zip(expected, vector_lines[1:])
    ) + "\n", encoding="utf-8")
    if kind == "ft_bin":
        import fasttext
        import numpy as np

        corpus = tmp_path / "corpus.txt"
        corpus.write_text((" ".join(expected) + "\n") * 100, encoding="utf-8")
        model = fasttext.train_unsupervised(str(corpus), dim=5, minCount=1, bucket=1000,
                                           epoch=0, thread=1, verbose=0)
        # The fixture needs a real binary model, not trained linguistic scores.
        # Set all word/subword vectors explicitly for deterministic finite input.
        rng = np.random.default_rng(13)
        model.set_matrices(
            rng.uniform(-0.1, 0.1, model.get_input_matrix().shape).astype(np.float32),
            np.zeros_like(model.get_output_matrix()),
        )
        vectors = tmp_path / "mini.bin"
        model.save_model(str(vectors))
    gold = tmp_path / "gold.csv"
    gold.write_text("word,score\n" + "\n".join(
        f"{word},{1 + index / 2}" for index, word in enumerate(words[:8])
    ) + "\n", encoding=gold_encoding)
    target = tmp_path / "target.txt"
    target.write_text("\n".join(words).replace("café", "cafe\u0301") + "\n", encoding="utf-8-sig")
    cfg["dataset"].update(gold=str(gold), lowercase=lowercase)
    cfg["embeddings"]["spaces"][0].update(path=str(vectors), kind=kind)
    cfg["runtime"]["output_dir"] = str(tmp_path / "output")
    cfg["prediction"]["target"] = str(target)
    outputs = PredictionPipeline().run_prediction(cfg, "unicode-test.json")
    with outputs["vocab_predictions"].open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["word"] for row in rows] == expected
    assert all(1 <= float(row["pred"]) <= 4.5 for row in rows)
    summary = json.loads(outputs["summary"].read_text(encoding="utf-8"))
    assert summary["gold_coverage"] == 1
    assert summary["n_vocab_rows"] == len(words)

    # Run the separate holdout command against the generated CSV as well.
    cfg["prediction"].update(holdout=str(gold), predictions_csv=str(outputs["vocab_predictions"]),
                              unmatched_csv=str(tmp_path / "unmatched.csv"))
    holdout_outputs = PredictionPipeline().run_holdout(cfg, "unicode-test.json")
    holdout_summary = json.loads(holdout_outputs["holdout_summary"].read_text(encoding="utf-8"))
    assert holdout_summary["n_matched"] == 8
    assert holdout_summary["overlap_ratio"] == 1


@pytest.mark.parametrize("suffix,delimiter", [(".csv", ","), (".tsv", "\t"), (".csv", "\t")])
def test_quoted_punctuation_survives_csv_and_delimiter_fallback(tmp_path, suffix, delimiter):
    path = tmp_path / ("gold" + suffix)
    words = ['quote"word', "comma,word", "<tag>", "a&b", "C++", "C#", "🙂", "inside\ufeffmarker"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle, delimiter=delimiter)
        writer.writerow(["word", "score"])
        writer.writerows((word, 2) for word in words)
    cfg = dataset(path)
    assert set(load_gold_df(cfg)["word"]) == set(words)
    assert inspect_gold({"dataset": cfg}, HostedLimits())["normalized_unique_rows"] == len(words)


@pytest.mark.parametrize("score", ["NaN", "inf", "-inf"])
def test_nonfinite_scores_are_rejected_by_loader_and_admission(tmp_path, score):
    path = tmp_path / "gold.csv"
    path.write_text(f"word,score\nNA,{score}\n", encoding="utf-8")
    cfg = dataset(path)
    with pytest.raises(ValueError, match="finite"):
        load_gold_df(cfg)
    with pytest.raises(ValueError, match="finite"):
        inspect_gold({"dataset": cfg}, HostedLimits())
