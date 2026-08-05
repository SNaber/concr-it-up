from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path

import pytest

from apps.core_gui.job_store import (
    JobAdmissionError,
    JobStoreError,
    SQLiteJobStore,
    utc_now,
)


def _config(output_dir: Path) -> dict:
    return {"runtime": {"output_dir": str(output_dir)}}


def _store(tmp_path: Path, **kwargs) -> SQLiteJobStore:
    return SQLiteJobStore(
        tmp_path / "queue.sqlite3",
        tmp_path / "jobs",
        **kwargs,
    )


def test_sqlite_wal_bundle_and_public_record(tmp_path: Path) -> None:
    store = _store(tmp_path)
    source = tmp_path / "source.csv"
    source.write_text("word,score\na,1\n", encoding="utf-8")
    job_id = store.new_job_id()
    paths = store.paths_for(job_id)

    submitted = store.submit(
        "prediction-run",
        {"config": {"live": True}},
        owner="session-a",
        config=_config(paths.output_dir),
        job_id=job_id,
        input_files={"gold.csv": source},
    )

    assert submitted == job_id
    assert store.journal_mode() == "wal"
    assert paths.input_dir.joinpath("gold.csv").read_text(encoding="utf-8") == source.read_text(
        encoding="utf-8"
    )
    assert json.loads(paths.config_path.read_text(encoding="utf-8"))["runtime"][
        "output_dir"
    ] == str(paths.output_dir)
    record = store.get(job_id, owner="session-a")
    assert record is not None
    assert record["status"] == "queued"
    assert record["payload"] == {"config": {"live": True}}
    assert isinstance(record["logs"], list)
    assert store.get(job_id, owner="session-b") is None


def test_atomic_claim_allows_only_one_global_running_job(tmp_path: Path) -> None:
    first_store = _store(tmp_path, global_queue_limit=3, per_owner_queue_limit=1)
    second_store = _store(tmp_path, global_queue_limit=3, per_owner_queue_limit=1)
    first = first_store.submit(
        "prediction-run",
        {},
        owner="one",
        config=_config(first_store.paths_for(first_store.new_job_id()).output_dir),
    )
    second = first_store.submit(
        "prediction-run",
        {},
        owner="two",
        config=_config(first_store.paths_for(first_store.new_job_id()).output_dir),
    )

    claimed = first_store.claim_next("worker-a")
    assert claimed is not None
    assert claimed.job_id in {first, second}
    assert second_store.claim_next("worker-b") is None
    assert first_store.heartbeat(claimed.job_id, "worker-a")
    first_store.complete(claimed.job_id, "worker-a", {"artifacts": {}})
    next_claim = second_store.claim_next("worker-b")
    assert next_claim is not None
    assert next_claim.job_id != claimed.job_id


def test_simultaneous_sqlite_claims_have_one_winner(tmp_path: Path) -> None:
    first_store = _store(tmp_path, global_queue_limit=3)
    second_store = _store(tmp_path, global_queue_limit=3)
    first_store.submit("prediction-run", {}, owner="one", config={})
    first_store.submit("prediction-run", {}, owner="two", config={})

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(first_store.claim_next, "worker-a"),
            pool.submit(second_store.claim_next, "worker-b"),
        ]
        claims = [future.result() for future in futures]

    winners = [claim for claim in claims if claim is not None]
    assert len(winners) == 1
    assert first_store.count_outstanding() == 2


def test_owner_and_global_queue_limits_are_transactional(tmp_path: Path) -> None:
    store = _store(tmp_path, global_queue_limit=2, per_owner_queue_limit=1)
    store.submit("prediction-run", {}, owner="a", config={})
    with pytest.raises(JobAdmissionError, match="Owner"):
        store.submit("prediction-run", {}, owner="a", config={})
    store.submit("prediction-run", {}, owner="b", config={})
    with pytest.raises(JobAdmissionError, match="Global"):
        store.submit("prediction-run", {}, owner="c", config={})


def test_active_owners_includes_every_active_state_and_excludes_terminal_jobs(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path, global_queue_limit=4)
    done_id = store.submit("prediction-run", {}, owner="done-owner", config={})
    running_id = store.submit("prediction-run", {}, owner="running-owner", config={})
    store.submit("prediction-run", {}, owner="queued-owner", config={})
    claimed = store.claim_next("worker")
    assert claimed is not None and claimed.job_id == done_id
    store.complete(done_id, "worker", {})
    claimed = store.claim_next("worker")
    assert claimed is not None and claimed.job_id == running_id

    preparing_paths = store.paths_for(store.new_job_id())
    store._reserve_row(
        paths=preparing_paths,
        command="prediction-run",
        payload={},
        owner="preparing-owner",
        timeout_seconds=60,
        output_limit_bytes=1_000,
    )

    assert store.active_owners() == {
        "preparing-owner",
        "queued-owner",
        "running-owner",
    }


def test_restart_marks_running_error_but_leaves_queued(tmp_path: Path) -> None:
    store = _store(tmp_path, global_queue_limit=3)
    running_id = store.submit("prediction-run", {}, owner="a", config={})
    queued_id = store.submit("prediction-run", {}, owner="b", config={})
    claimed = store.claim_next("old-worker")
    assert claimed is not None and claimed.job_id == running_id

    recovered = store.recover_interrupted_jobs()

    assert recovered == [running_id]
    assert store.get(running_id)["status"] == "error"
    assert store.get(queued_id)["status"] == "queued"
    assert "Worker restarted" in store.get(running_id)["error"]


def test_lease_is_required_for_completion_and_failure(tmp_path: Path) -> None:
    store = _store(tmp_path)
    job_id = store.submit("prediction-run", {}, owner="a", config={})
    claimed = store.claim_next("right-worker")
    assert claimed is not None
    with pytest.raises(JobStoreError, match="active lease"):
        store.complete(job_id, "wrong-worker", {})
    with pytest.raises(JobStoreError, match="active state/lease"):
        store.fail(job_id, "no", worker_id="wrong-worker")


def test_disk_admission_limit_rejects_before_creating_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(tmp_path)
    monkeypatch.setattr(store, "admission_available", lambda: False)
    with pytest.raises(JobAdmissionError, match="admission threshold"):
        store.submit("prediction-run", {}, owner="a", config={})
    assert list((tmp_path / "jobs").iterdir()) == []


@pytest.mark.parametrize(
    ("used_fraction", "expected"),
    [(0.79, True), (0.80, False)],
)
def test_storage_admission_threshold_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    used_fraction: float,
    expected: bool,
) -> None:
    store = _store(tmp_path, storage_stop_fraction=0.80)
    monkeypatch.setattr(
        store,
        "storage_usage",
        lambda: {"used_fraction": used_fraction},
    )

    assert store.admission_available() is expected


def test_cleanup_removes_only_expired_terminal_job_bundle(tmp_path: Path) -> None:
    store = _store(tmp_path, global_queue_limit=3)
    old_id = store.submit("prediction-run", {}, owner="a", config={})
    keep_id = store.submit("prediction-run", {}, owner="b", config={})
    old_claim = store.claim_next("worker")
    assert old_claim is not None and old_claim.job_id == old_id
    store.complete(old_id, "worker", {})
    past = (utc_now() - timedelta(hours=72)).isoformat()
    with store._connect() as connection:
        connection.execute(
            "UPDATE jobs SET finished_at=?, updated_at=? WHERE job_id=?",
            (past, past, old_id),
        )

    removed = store.cleanup_expired(retention_hours=48)

    assert removed == [old_id]
    assert not store.paths_for(old_id).job_dir.exists()
    assert store.get(old_id) is None
    assert store.get(keep_id)["status"] == "queued"


def test_existing_result_registration_is_signature_idempotent_and_owner_scoped(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    kwargs = {
        "command": "existing-results",
        "payload": {"source": "paper"},
        "result": {"artifacts": {"summary": "summary.json"}},
        "signature": "stable",
        "owner": "session-a",
    }
    first = store.register_existing_result(**kwargs)
    second = store.register_existing_result(**kwargs)
    assert first == second
    other_owner = store.register_existing_result(**{**kwargs, "owner": "session-b"})
    assert other_owner != first
    assert store.latest_completed_job(owner="session-a")["job_id"] == first
    assert store.latest_completed_job(owner="session-b")["job_id"] == other_owner
