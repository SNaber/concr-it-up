import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from concreteness_knn_core.prediction import PredictionPipeline
from concreteness_knn_core.config import load_config, validate_config


class _DummyEmbeddings:
    def __init__(self, offset: int = 0):
        self._offset = int(offset)

    def get_vector(self, word: str):
        base = float((sum(ord(c) for c in word) + self._offset) % 19) / 19.0
        return np.asarray([base, base + 0.1, base + 0.2, base + 0.3], dtype=np.float32)


class TestPredictionPipeline(unittest.TestCase):
    def test_prediction_run_writes_expected_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            words = [f"w{i}" for i in range(16)]
            gold = tmp / "gold.csv"
            pd.DataFrame({"Word": words, "Conc.M": [1.0 + (i % 5) for i in range(16)]}).to_csv(
                gold, index=False
            )

            vocab_txt = tmp / "vocab.txt"
            vocab_txt.write_text("\n".join(words[:6]), encoding="utf-8")

            cfg_path = tmp / "cfg.json"
            payload = {
                "dataset": {
                    "gold": str(gold),
                    "word_column": "Word",
                    "score_column": "Conc.M",
                    "lowercase": True,
                    "pos_filter": {"enabled": False, "pos_column": "Dom_Pos", "tags": ["Noun"]},
                },
                "embeddings": {
                    "mode": "single",
                    "active_space": "dummy",
                    "spaces": [{"id": "dummy", "kind": "vec", "path": "dummy.vec", "label": "dummy"}],
                },
                "runtime": {"output_dir": str(tmp / "out"), "seed": 13, "n_jobs": 1},
                "reports": {"level": "full"},
                "prediction": {
                    "test_size": 0.25,
                    "cv_folds": 2,
                    "k_min": 1,
                    "k_max": 3,
                    "k_step": 1,
                    "topn_neighbors": 3,
                    "target": str(vocab_txt),
                },
            }
            cfg_path.write_text(json.dumps(payload), encoding="utf-8")
            cfg = load_config(str(cfg_path))
            validate_config(cfg, task="prediction_run")

            with patch(
                "concreteness_knn_core.prediction.prediction_data_prep.load_embeddings_by_space",
                return_value={"dummy": _DummyEmbeddings()},
            ):
                outputs = PredictionPipeline().run_prediction(cfg, config_path=str(cfg_path))

            expected = {
                "summary",
                "cv_results",
                "test_predictions",
                "vocab_predictions",
                "oov_gold",
                "oov_vocab",
            }
            self.assertTrue(expected.issubset(set(outputs.keys())))
            for key in expected:
                self.assertTrue(Path(outputs[key]).exists())
            self.assertEqual(Path(outputs["summary"]).name, "summary.json")
            self.assertEqual(Path(outputs["cv_results"]).name, "cv_results.csv")
            self.assertEqual(Path(outputs["test_predictions"]).name, "test_predictions.csv")
            self.assertEqual(Path(outputs["vocab_predictions"]).name, "vocab_predictions.csv")
            self.assertEqual(Path(outputs["oov_gold"]).name, "oov_gold.csv")
            self.assertEqual(Path(outputs["oov_vocab"]).name, "oov_vocab.csv")

            summary = json.loads(Path(outputs["summary"]).read_text(encoding="utf-8"))
            self.assertIn("schema_version", summary)
            self.assertEqual(summary["vocab_fit_scope"], "all_covered_gold_post_eval")
            self.assertEqual(int(summary["n_vocab_reference_words"]), int(summary["n_gold_with_embeddings"]))

            test_df = pd.read_csv(outputs["test_predictions"])
            required_cols = {"word", "gold", "pred", "error", "abs_error", "pred__dummy"}
            self.assertTrue(required_cols.issubset(set(test_df.columns)))

            vocab_df = pd.read_csv(outputs["vocab_predictions"])
            self.assertIn("is_gold_observed", vocab_df.columns)

    def test_vocab_predictions_exclude_self_neighbors_and_recompute_pred(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            words = [f"w{i}" for i in range(24)]
            gold = tmp / "gold.csv"
            pd.DataFrame({"Word": words, "Conc.M": [1.0 + (i % 7) for i in range(24)]}).to_csv(
                gold, index=False
            )

            vocab_txt = tmp / "vocab.txt"
            vocab_txt.write_text("\n".join(words), encoding="utf-8")

            cfg_path = tmp / "cfg.json"
            payload = {
                "dataset": {
                    "gold": str(gold),
                    "word_column": "Word",
                    "score_column": "Conc.M",
                    "lowercase": True,
                    "pos_filter": {"enabled": False, "pos_column": "Dom_Pos", "tags": ["Noun"]},
                },
                "embeddings": {
                    "mode": "single",
                    "active_space": "dummy",
                    "spaces": [{"id": "dummy", "kind": "vec", "path": "dummy.vec", "label": "dummy"}],
                },
                "runtime": {"output_dir": str(tmp / "out"), "seed": 13, "n_jobs": 1},
                "reports": {"level": "full"},
                "prediction": {
                    "test_size": 0.25,
                    "cv_folds": 3,
                    "k_min": 1,
                    "k_max": 3,
                    "k_step": 1,
                    "topn_neighbors": 3,
                    "target": str(vocab_txt),
                },
            }
            cfg_path.write_text(json.dumps(payload), encoding="utf-8")
            cfg = load_config(str(cfg_path))
            validate_config(cfg, task="prediction_run")

            with patch(
                "concreteness_knn_core.prediction.prediction_data_prep.load_embeddings_by_space",
                return_value={"dummy": _DummyEmbeddings()},
            ):
                outputs = PredictionPipeline().run_prediction(cfg, config_path=str(cfg_path))

            summary = json.loads(Path(outputs["summary"]).read_text(encoding="utf-8"))
            self.assertEqual(int(summary["n_vocab_rows"]), 24)
            self.assertGreaterEqual(int(summary["n_self_filtered_rows"]), 1)
            self.assertGreater(float(summary["self_filtered_ratio"]), 0.0)
            self.assertEqual(summary["vocab_fit_scope"], "all_covered_gold_post_eval")
            self.assertEqual(int(summary["n_vocab_reference_words"]), int(summary["n_gold_with_embeddings"]))

            test_df = pd.read_csv(outputs["test_predictions"])
            test_words = set(test_df["word"].astype(str))

            vocab_df = pd.read_csv(outputs["vocab_predictions"])
            heldout_overlap = vocab_df[vocab_df["word"].isin(test_words)]
            self.assertFalse(heldout_overlap.empty)
            self.assertTrue(
                heldout_overlap["is_gold_observed"].astype(str).str.lower().eq("true").all()
            )

            for _, row in heldout_overlap.iterrows():
                neighbors = [token.strip() for token in str(row["neighbors__dummy"]).split("|") if token.strip()]
                self.assertNotIn(str(row["word"]), neighbors)

            probe = heldout_overlap.iloc[0]
            scores = [
                float(token.strip())
                for token in str(probe["neighbor_gold_scores__dummy"]).split("|")
                if token.strip()
            ]
            distances = [
                float(token.strip())
                for token in str(probe["neighbor_cosine_distances__dummy"]).split("|")
                if token.strip()
            ]
            k = min(int(summary["best_k"]), len(scores))
            self.assertGreaterEqual(k, 1)

            if str(summary["best_weights"]) == "uniform":
                expected = float(np.mean(scores[:k]))
            else:
                w = 1.0 / np.clip(np.asarray(distances[:k], dtype=np.float32), 1e-12, None)
                expected = float((w * np.asarray(scores[:k], dtype=np.float32)).sum() / np.clip(w.sum(), 1e-12, None))
            self.assertAlmostEqual(float(probe["pred__dummy"]), expected, places=6)

    def test_validation_rejects_invalid_ranges(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            cfg_path = tmp / "cfg.json"
            payload = {
                "dataset": {
                    "gold": str(tmp / "gold.csv"),
                    "word_column": "Word",
                    "score_column": "Conc.M",
                    "lowercase": True,
                    "pos_filter": {"enabled": False, "pos_column": "Dom_Pos", "tags": ["Noun"]},
                },
                "embeddings": {
                    "mode": "single",
                    "active_space": "dummy",
                    "spaces": [{"id": "dummy", "kind": "vec", "path": "dummy.vec", "label": "dummy"}],
                },
                "runtime": {"output_dir": str(tmp / "out"), "seed": 13, "n_jobs": 1},
                "prediction": {
                    "test_size": 1.0,
                    "cv_folds": 2,
                    "k_min": 1,
                    "k_max": 3,
                    "k_step": 1,
                    "topn_neighbors": 3,
                },
            }
            cfg_path.write_text(json.dumps(payload), encoding="utf-8")
            cfg = load_config(str(cfg_path))
            with self.assertRaisesRegex(ValueError, "prediction.test_size"):
                validate_config(cfg, task="prediction_run")

    def test_prediction_joint_mode_outputs_joint_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            words = [f"w{i}" for i in range(16)]
            gold = tmp / "gold.csv"
            pd.DataFrame({"Word": words, "Conc.M": [1.0 + (i % 5) for i in range(16)]}).to_csv(
                gold, index=False
            )

            cfg_path = tmp / "cfg.json"
            payload = {
                "dataset": {
                    "gold": str(gold),
                    "word_column": "Word",
                    "score_column": "Conc.M",
                    "lowercase": True,
                    "pos_filter": {"enabled": False, "pos_column": "Dom_Pos", "tags": ["Noun"]},
                },
                "embeddings": {
                    "mode": "joint",
                    "spaces": [
                        {"id": "s1", "kind": "vec", "path": "s1.vec", "label": "s1"},
                        {"id": "s2", "kind": "vec", "path": "s2.vec", "label": "s2"},
                    ],
                },
                "runtime": {"output_dir": str(tmp / "out"), "seed": 13, "n_jobs": 1},
                "prediction": {
                    "test_size": 0.25,
                    "cv_folds": 2,
                    "k_min": 1,
                    "k_max": 3,
                    "k_step": 1,
                    "topn_neighbors": 3,
                },
            }
            cfg_path.write_text(json.dumps(payload), encoding="utf-8")
            cfg = load_config(str(cfg_path))
            validate_config(cfg, task="prediction_run")

            with patch(
                "concreteness_knn_core.prediction.prediction_data_prep.load_embeddings_by_space",
                return_value={"s1": _DummyEmbeddings(0), "s2": _DummyEmbeddings(3)},
            ):
                outputs = PredictionPipeline().run_prediction(cfg, config_path=str(cfg_path))

            test_df = pd.read_csv(outputs["test_predictions"])
            self.assertIn("pred__joint", test_df.columns)

    def test_prediction_multi_space_outputs_per_space_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            words = [f"w{i}" for i in range(16)]
            gold = tmp / "gold.csv"
            pd.DataFrame({"Word": words, "Conc.M": [1.0 + (i % 5) for i in range(16)]}).to_csv(
                gold, index=False
            )

            cfg_path = tmp / "cfg.json"
            payload = {
                "dataset": {
                    "gold": str(gold),
                    "word_column": "Word",
                    "score_column": "Conc.M",
                    "lowercase": True,
                    "pos_filter": {"enabled": False, "pos_column": "Dom_Pos", "tags": ["Noun"]},
                },
                "embeddings": {
                    "mode": "multi_space",
                    "spaces": [
                        {"id": "s1", "kind": "vec", "path": "s1.vec", "label": "s1"},
                        {"id": "s2", "kind": "vec", "path": "s2.vec", "label": "s2"},
                    ],
                },
                "runtime": {"output_dir": str(tmp / "out"), "seed": 13, "n_jobs": 1},
                "prediction": {
                    "test_size": 0.25,
                    "cv_folds": 2,
                    "k_min": 1,
                    "k_max": 3,
                    "k_step": 1,
                    "topn_neighbors": 3,
                },
            }
            cfg_path.write_text(json.dumps(payload), encoding="utf-8")
            cfg = load_config(str(cfg_path))
            validate_config(cfg, task="prediction_run")

            with patch(
                "concreteness_knn_core.prediction.prediction_data_prep.load_embeddings_by_space",
                return_value={"s1": _DummyEmbeddings(0), "s2": _DummyEmbeddings(3)},
            ):
                outputs = PredictionPipeline().run_prediction(cfg, config_path=str(cfg_path))

            test_df = pd.read_csv(outputs["test_predictions"])
            self.assertIn("pred__s1", test_df.columns)
            self.assertIn("pred__s2", test_df.columns)


if __name__ == "__main__":
    unittest.main()
