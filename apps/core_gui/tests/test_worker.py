from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from apps.core_gui.job_store import SQLiteJobStore
from apps.core_gui.hosted_config import owner_storage_key
from apps.core_gui.session_storage import ACTIVITY_FILENAME, acquire_session_storage_lease
from apps.core_gui.worker import JobWorker, WorkerSettings


def _settings(tmp_path: Path, *, entrypoint: str = "fake-core") -> WorkerSettings:
    return WorkerSettings(
        repo_root=tmp_path,
        db_path=tmp_path / "queue.sqlite3",
        job_root=tmp_path / "jobs",
        entrypoint=entrypoint,
        worker_id="test-worker",
        session_root=tmp_path / "sessions",
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


def test_worker_settings_wires_validated_session_root(tmp_path: Path, monkeypatch) -> None:
    session_root = tmp_path / "data" / "owner_sessions"
    monkeypatch.setenv("CONCRITUP_REPO_ROOT", str(tmp_path))
    monkeypatch.setenv("CONCRITUP_SESSION_ROOT", str(session_root))
    monkeypatch.setenv("CONCRITUP_RETENTION_HOURS", "36.5")

    settings = WorkerSettings.from_env()

    assert settings.session_root == session_root.resolve(strict=False)
    assert settings.retention_hours == 36.5


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


def test_worker_cleanup_removes_inactive_session_but_preserves_active_owner(
    tmp_path: Path,
    monkeypatch,
) -> None:
    settings = _settings(tmp_path)
    store = SQLiteJobStore(settings.db_path, settings.job_root, global_queue_limit=3)
    active_owner = "active-owner"
    store.submit("prediction-run", {}, owner=active_owner, config={})

    reference = datetime.now(timezone.utc)
    inactive_key = owner_storage_key("inactive-owner")
    active_key = owner_storage_key(active_owner)
    for key in (inactive_key, active_key):
        with acquire_session_storage_lease(settings.session_root, key):
            marker = settings.session_root / key / ACTIVITY_FILENAME
            timestamp = (reference - timedelta(days=7)).timestamp()
            os.utime(marker, (timestamp, timestamp))

    monotonic = iter((100.0, 399.0, 400.0))
    monkeypatch.setattr("apps.core_gui.worker.time.monotonic", lambda: next(monotonic))
    worker = JobWorker(store, settings)

    first = worker.cleanup_if_due()
    assert first == [f"session:{inactive_key}"]
    assert not (settings.session_root / inactive_key).exists()
    assert (settings.session_root / active_key).is_dir()

    assert worker.cleanup_if_due() == []

    second_key = owner_storage_key("second-inactive")
    with acquire_session_storage_lease(settings.session_root, second_key):
        marker = settings.session_root / second_key / ACTIVITY_FILENAME
        timestamp = (reference - timedelta(days=7)).timestamp()
        os.utime(marker, (timestamp, timestamp))
    assert worker.cleanup_if_due() == [f"session:{second_key}"]


def test_forced_cleanup_runs_job_and_session_retention_in_one_cycle(
    tmp_path: Path,
    monkeypatch,
) -> None:
    settings = _settings(tmp_path)
    store = SQLiteJobStore(settings.db_path, settings.job_root, global_queue_limit=3)
    monkeypatch.setattr(store, "cleanup_expired", lambda **_kwargs: ["expired-job"])
    calls: list[dict] = []

    def fake_session_cleanup(_root, **kwargs):
        calls.append(kwargs)
        assert callable(kwargs["active_storage_keys_provider"])
        assert kwargs["active_storage_keys_provider"]() == set()
        return [owner_storage_key("expired-owner")]

    monkeypatch.setattr(
        "apps.core_gui.worker.cleanup_expired_session_storage",
        fake_session_cleanup,
    )
    monkeypatch.setattr("apps.core_gui.worker.time.monotonic", lambda: 100.0)
    worker = JobWorker(store, settings)

    expected = ["expired-job", f"session:{owner_storage_key('expired-owner')}"]
    assert worker.cleanup_if_due() == expected
    assert worker.cleanup_if_due() == []
    assert worker.cleanup_if_due(force=True) == expected
    assert len(calls) == 2
    assert all(call["retention_hours"] == 48 for call in calls)
