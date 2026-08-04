"""End-to-end test for the self-contained synthetic example."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_DIR = REPO_ROOT / "examples" / "smoke"


def test_synthetic_prediction_and_holdout(tmp_path: Path) -> None:
    payload = json.loads((EXAMPLE_DIR / "config.json").read_text(encoding="utf-8"))
    output_dir = tmp_path / "output"

    payload["dataset"]["gold"] = str(EXAMPLE_DIR / "gold.csv")
    payload["embeddings"]["spaces"][0]["path"] = str(EXAMPLE_DIR / "mini.vec")
    payload["runtime"]["output_dir"] = str(output_dir)
    payload["prediction"]["target"] = str(EXAMPLE_DIR / "target.txt")
    payload["prediction"]["holdout"] = str(EXAMPLE_DIR / "holdout.csv")
    payload["prediction"]["predictions_csv"] = str(output_dir / "vocab_predictions.csv")
    payload["prediction"]["unmatched_csv"] = str(output_dir / "holdout_unmatched.csv")

    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    prediction = subprocess.run(
        [
            sys.executable,
            "-m",
            "concreteness_knn_core",
            "prediction",
            "run",
            "--config",
            str(config_path),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert prediction.returncode == 0, prediction.stdout + prediction.stderr

    expected_prediction = {
        "summary.json",
        "cv_results.csv",
        "test_predictions.csv",
        "vocab_predictions.csv",
        "oov_gold.csv",
        "oov_vocab.csv",
    }
    assert expected_prediction.issubset({path.name for path in output_dir.iterdir()})

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["n_gold_words"] == 8
    assert summary["n_vocab_rows"] == 4
    assert summary["gold_coverage"] == 1.0
    assert summary["vocab_fit_scope"] == "all_covered_gold_post_eval"

    holdout = subprocess.run(
        [
            sys.executable,
            "-m",
            "concreteness_knn_core",
            "prediction",
            "holdout",
            "--config",
            str(config_path),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert holdout.returncode == 0, holdout.stdout + holdout.stderr
    assert (output_dir / "holdout_summary.json").exists()
    assert (output_dir / "holdout_unmatched.csv").exists()

    holdout_summary = json.loads(
        (output_dir / "holdout_summary.json").read_text(encoding="utf-8")
    )
    assert holdout_summary["n_matched"] == 4
    assert holdout_summary["overlap_ratio"] == 1.0
