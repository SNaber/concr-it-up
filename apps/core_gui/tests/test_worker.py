from __future__ import annotations

import json
import os
from pathlib import Path

from apps.core_gui.job_store import SQLiteJobStore
from apps.core_gui.worker import JobWorker, WorkerSettings


def _settings(tmp_path: Path, *, entrypoint: str = "fake-core") -> WorkerSettings:
    return WorkerSettings(
        repo_root=tmp_path,
        db_path=tmp_path / "queue.sqlite3",
        job_root=tmp_path / "jobs",
        entrypoint=entrypoint,
        worker_id="test-worker",
        poll_seconds=0.05,
        lease_seconds=3,
        retention_hours=48,
        cleanup_interval_seconds=300,
    )


def _config(paths, gold: Path, embedding: Path) -> dict:
    return {
        "dataset": {
            "gold": str(gold),
            "word_column": "word",
            "score_column": "score",
            "lowercase": True,
            "pos_filter": {"enabled": False, "tags": []},
        },
        "embeddings": {
            "mode": "single",
            "active_space": "mini",
            "spaces": [
                {"id": "mini", "kind": "vec", "path": str(embedding), "label": "mini"}
            ],
        },
        "runtime": {"output_dir": str(paths.output_dir), "n_jobs": 1, "seed": 13},
        "prediction": {"target": None, "holdout": None},
        "reports": {"level": "core"},
    }


def _submitted(tmp_path: Path, *, timeout: int = 60, output_limit: int = 1_000_000):
    settings = _settings(tmp_path)
    store = SQLiteJobStore(
        settings.db_path,
        settings.job_root,
        global_queue_limit=20,
        redact_roots=(tmp_path,),
    )
    gold = tmp_path / "gold.csv"
    embedding = tmp_path / "mini.vec"
    gold.write_text("word,score\na,1\n", encoding="utf-8")
    embedding.write_text("1 2\na 1 0\n", encoding="utf-8")
    job_id = store.new_job_id()
    paths = store.paths_for(job_id)
    store.submit(
        "prediction-run",
        {"config": "editor payload"},
        owner="opaque-session-token",
        config=_config(paths, gold, embedding),
        job_id=job_id,
        timeout_seconds=timeout,
        output_limit_bytes=output_limit,
    )
    return settings, store, job_id, paths


def _write_prediction_artifacts(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps({"best_k": 5, "config_path": "/private/server/config.json"}),
        encoding="utf-8",
    )
    (output_dir / "cv_results.csv").write_text("k,weights\n5,distance\n", encoding="utf-8")
    (output_dir / "test_predictions.csv").write_text("word,pred\na,1\n", encoding="utf-8")


def test_worker_success_writes_manifest_and_completes(tmp_path: Path) -> None:
    settings, store, job_id, paths = _submitted(tmp_path)

    def executor(job, command, cwd, environment) -> int:
        assert command[-1] == str(paths.config_path)
        assert cwd == tmp_path
        assert environment["OMP_NUM_THREADS"] == "1"
        _write_prediction_artifacts(job.paths.output_dir)
        store.append_log(job.job_id, "core output with " + str(tmp_path))
        return 0

    worker = JobWorker(store, settings, executor=executor)
    assert worker.run_once()

    record = store.get(job_id)
    assert record["status"] == "done"
    assert set(record["result"]["artifacts"]) >= {
        "summary",
        "cv_results",
        "test_predictions",
        "run_manifest",
    }
    manifest = json.loads(paths.manifest_path.read_text(encoding="utf-8"))
    rendered = json.dumps(manifest)
    assert "opaque-session-token" not in rendered
    assert str(tmp_path) not in rendered
    assert manifest["checksums"]["inputs"]["gold"]["sha256"]
    assert manifest["checksums"]["embeddings"]["mini"]["sha256"]
    assert manifest["normalization"]["unicode"].startswith("No Unicode normalization")
    assert manifest["command_argv"][-1] == "job/config.json"
    assert any("<server-path>" in line for line in record["logs"])


def test_worker_nonzero_subprocess_result_is_public_error(tmp_path: Path) -> None:
    settings, store, job_id, paths = _submitted(tmp_path)
    worker = JobWorker(store, settings, executor=lambda *_args: 7)

    assert worker.run_once()

    record = store.get(job_id)
    assert record["status"] == "error"
    assert record["error"] == "Model subprocess exited with status 7."
    manifest = json.loads(paths.manifest_path.read_text(encoding="utf-8"))
    assert manifest["job"]["status"] == "error"
    assert manifest["job"]["return_code"] == 7


def test_worker_detects_output_limit_after_injected_executor(tmp_path: Path) -> None:
    settings, store, job_id, paths = _submitted(tmp_path, output_limit=10)

    def executor(job, *_args) -> int:
        (job.paths.output_dir / "large.bin").write_bytes(b"x" * 11)
        return 0

    assert JobWorker(store, settings, executor=executor).run_once()
    record = store.get(job_id)
    assert record["status"] == "error"
    assert "output exceeded" in record["error"]


def test_real_subprocess_timeout_and_combined_log_capture(tmp_path: Path) -> None:
    script = tmp_path / "slow-core"
    script.write_text(
        "#!/usr/bin/env python3\n"
        "import sys, time\n"
        "print('subprocess-started', flush=True)\n"
        "time.sleep(5)\n",
        encoding="utf-8",
    )
    script.chmod(0o755)
    settings, store, job_id, paths = _submitted(tmp_path, timeout=1)
    settings = WorkerSettings(**{**settings.__dict__, "entrypoint": str(script)})

    assert JobWorker(store, settings).run_once()

    record = store.get(job_id)
    assert record["status"] == "error"
    assert "timeout" in record["error"]
    assert "subprocess-started" in paths.log_path.read_text(encoding="utf-8")


def test_worker_environment_forces_single_thread_even_if_parent_is_larger(monkeypatch) -> None:
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "32")
    environment = JobWorker.subprocess_environment()
    assert environment["OPENBLAS_NUM_THREADS"] == "1"
    assert environment["MKL_NUM_THREADS"] == "1"
    assert environment["SKLEARN_WORKING_MEMORY"] == "256"
