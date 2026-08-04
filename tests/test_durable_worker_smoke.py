from __future__ import annotations

import json
import sys
from pathlib import Path

from apps.core_gui.job_store import SQLiteJobStore
from apps.core_gui.worker import JobWorker, WorkerSettings


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_ROOT = REPO_ROOT / "examples" / "smoke"


def _job_config(output_dir: Path) -> dict:
    config = json.loads((EXAMPLE_ROOT / "config.json").read_text(encoding="utf-8"))
    config["dataset"]["gold"] = str(EXAMPLE_ROOT / "gold.csv")
    config["embeddings"]["spaces"][0]["path"] = str(EXAMPLE_ROOT / "mini.vec")
    config["runtime"]["output_dir"] = str(output_dir)
    config["runtime"]["n_jobs"] = 1
    config["prediction"]["target"] = str(EXAMPLE_ROOT / "target.txt")
    config["prediction"]["holdout"] = str(EXAMPLE_ROOT / "holdout.csv")
    config["prediction"]["predictions_csv"] = str(output_dir / "vocab_predictions.csv")
    config["prediction"]["unmatched_csv"] = str(output_dir / "holdout_unmatched.csv")
    return config


def test_durable_worker_runs_fresh_prediction_and_holdout_subprocesses(tmp_path: Path) -> None:
    entrypoint = Path(sys.executable).parent / "concreteness-knn-core"
    assert entrypoint.is_file(), f"Missing installed core entrypoint: {entrypoint}"
    store = SQLiteJobStore(tmp_path / "jobs.sqlite3", tmp_path / "jobs")
    settings = WorkerSettings(
        repo_root=REPO_ROOT,
        db_path=store.db_path,
        job_root=store.job_root,
        entrypoint=str(entrypoint),
        worker_id="smoke-worker",
        poll_seconds=0.05,
    )
    worker = JobWorker(store, settings)

    prediction_id = store.new_job_id()
    prediction_paths = store.paths_for(prediction_id)
    prediction_config = _job_config(prediction_paths.output_dir)
    store.submit(
        "prediction-run",
        {"config": "synthetic-smoke"},
        owner="smoke-session",
        config=prediction_config,
        job_id=prediction_id,
        timeout_seconds=60,
        output_limit_bytes=10 * 1024 * 1024,
    )
    assert worker.run_once() is True
    prediction = store.get(prediction_id)
    assert prediction["status"] == "done", prediction
    assert set(prediction["result"]["artifacts"]) >= {
        "summary",
        "cv_results",
        "test_predictions",
        "vocab_predictions",
        "oov_gold",
        "oov_vocab",
        "run_manifest",
    }

    holdout_id = store.new_job_id()
    holdout_paths = store.paths_for(holdout_id)
    holdout_config = _job_config(holdout_paths.output_dir)
    holdout_config["prediction"]["predictions_csv"] = prediction["result"]["artifacts"][
        "vocab_predictions"
    ]
    store.submit(
        "prediction-holdout",
        {"config": "synthetic-smoke"},
        owner="smoke-session",
        config=holdout_config,
        job_id=holdout_id,
        timeout_seconds=60,
        output_limit_bytes=10 * 1024 * 1024,
    )
    assert worker.run_once() is True
    holdout = store.get(holdout_id)
    assert holdout["status"] == "done", holdout
    assert set(holdout["result"]["artifacts"]) >= {
        "holdout_summary",
        "holdout_unmatched",
        "run_manifest",
    }
    summary = json.loads(holdout_paths.output_dir.joinpath("holdout_summary.json").read_text())
    assert summary["n_matched"] == 4
    assert summary["overlap_ratio"] == 1.0

