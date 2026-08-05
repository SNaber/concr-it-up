from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from apps.core_gui.hosted_config import owner_storage_key
from apps.core_gui.session_storage import (
    ACTIVITY_FILENAME,
    LOCK_DIRECTORY,
    SessionStorageError,
    TRASH_DIRECTORY,
    acquire_session_storage_lease,
    cleanup_expired_session_storage,
)


REFERENCE_TIME = datetime(2026, 8, 5, 12, 0, tzinfo=timezone.utc)


def _owner_tree(session_root: Path, owner: str) -> tuple[str, Path]:
    key = owner_storage_key(owner)
    with acquire_session_storage_lease(session_root, key):
        owner_root = session_root / key
        (owner_root / "uploads").mkdir(exist_ok=True)
        (owner_root / "configs").mkdir(exist_ok=True)
        (owner_root / "uploads" / "gold.csv").write_text("word,score\na,1\n")
        (owner_root / "configs" / "run.json").write_text("{}\n")
    return key, owner_root


def _set_activity(owner_root: Path, when: datetime) -> None:
    timestamp = when.timestamp()
    os.utime(owner_root / ACTIVITY_FILENAME, (timestamp, timestamp))


def test_session_cleanup_enforces_boundary_and_gives_legacy_grace(tmp_path: Path) -> None:
    session_root = tmp_path / "sessions"
    old_key, old_root = _owner_tree(session_root, "old")
    exact_key, exact_root = _owner_tree(session_root, "exact")
    fresh_key, fresh_root = _owner_tree(session_root, "fresh")
    _set_activity(old_root, REFERENCE_TIME - timedelta(hours=49))
    _set_activity(exact_root, REFERENCE_TIME - timedelta(hours=48))
    _set_activity(fresh_root, REFERENCE_TIME - timedelta(hours=47))

    legacy_key = owner_storage_key("legacy")
    legacy_root = session_root / legacy_key
    (legacy_root / "uploads").mkdir(parents=True)
    (legacy_root / "uploads" / "legacy.csv").write_text("legacy\n")

    removed = cleanup_expired_session_storage(
        session_root,
        retention_hours=48,
        now=REFERENCE_TIME,
    )

    assert set(removed) == {old_key, exact_key}
    assert not old_root.exists()
    assert not exact_root.exists()
    assert fresh_root.is_dir()
    assert legacy_root.is_dir()
    assert (legacy_root / ACTIVITY_FILENAME).is_file()
    assert (legacy_root / ACTIVITY_FILENAME).stat().st_mtime == pytest.approx(
        REFERENCE_TIME.timestamp()
    )
    assert (session_root / LOCK_DIRECTORY).is_dir()
    assert (session_root / TRASH_DIRECTORY).is_dir()

    assert cleanup_expired_session_storage(
        session_root,
        active_storage_keys={fresh_key},
        retention_hours=48,
        now=REFERENCE_TIME + timedelta(hours=47, minutes=59),
    ) == []
    assert legacy_root.is_dir()
    assert cleanup_expired_session_storage(
        session_root,
        active_storage_keys={fresh_key},
        retention_hours=48,
        now=REFERENCE_TIME + timedelta(hours=48),
    ) == [legacy_key]
    assert not legacy_root.exists()


def test_active_owner_and_shared_request_lock_defer_cleanup(tmp_path: Path) -> None:
    session_root = tmp_path / "sessions"
    active_key, active_root = _owner_tree(session_root, "active")
    locked_key, locked_root = _owner_tree(session_root, "locked")
    _set_activity(active_root, REFERENCE_TIME - timedelta(days=7))

    lease = acquire_session_storage_lease(session_root, locked_key)
    try:
        _set_activity(locked_root, REFERENCE_TIME - timedelta(days=7))
        assert cleanup_expired_session_storage(
            session_root,
            active_storage_keys={active_key},
            retention_hours=48,
            now=REFERENCE_TIME,
        ) == []
        assert active_root.is_dir()
        assert locked_root.is_dir()
    finally:
        lease.close()

    assert cleanup_expired_session_storage(
        session_root,
        active_storage_keys={active_key},
        retention_hours=48,
        now=REFERENCE_TIME,
    ) == [locked_key]
    assert active_root.is_dir()
    assert not locked_root.exists()


def test_cleanup_skips_unsafe_entries_and_never_follows_symlinks(tmp_path: Path) -> None:
    session_root = tmp_path / "sessions"
    session_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "keep.txt"
    sentinel.write_text("keep\n")

    symlink_key = owner_storage_key("symlink-root")
    (session_root / symlink_key).symlink_to(outside, target_is_directory=True)

    nested_key, nested_root = _owner_tree(session_root, "nested-symlink")
    (nested_root / "uploads" / "outside-link").symlink_to(
        outside,
        target_is_directory=True,
    )
    _set_activity(nested_root, REFERENCE_TIME - timedelta(days=7))

    invalid_dir = session_root / "not-an-owner"
    invalid_dir.mkdir()
    regular_key = owner_storage_key("regular-file")
    (session_root / regular_key).write_text("not a directory\n")

    removed = cleanup_expired_session_storage(
        session_root,
        retention_hours=48,
        now=REFERENCE_TIME,
    )

    assert removed == [nested_key]
    assert sentinel.read_text() == "keep\n"
    assert (session_root / symlink_key).is_symlink()
    assert invalid_dir.is_dir()
    assert (session_root / regular_key).is_file()


def test_final_active_provider_is_checked_under_exclusive_lock(tmp_path: Path) -> None:
    session_root = tmp_path / "sessions"
    key, owner_root = _owner_tree(session_root, "became-active")
    _set_activity(owner_root, REFERENCE_TIME - timedelta(days=7))
    calls = 0

    def became_active() -> set[str]:
        nonlocal calls
        calls += 1
        return {key}

    assert cleanup_expired_session_storage(
        session_root,
        active_storage_keys=(),
        active_storage_keys_provider=became_active,
        retention_hours=48,
        now=REFERENCE_TIME,
    ) == []
    assert calls == 1
    assert owner_root.is_dir()


def test_cleanup_rejects_a_symlinked_root_without_touching_target(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "keep.txt"
    sentinel.write_text("keep\n")
    session_root = tmp_path / "sessions"
    session_root.symlink_to(outside, target_is_directory=True)

    with pytest.raises(SessionStorageError, match="not a real directory"):
        cleanup_expired_session_storage(
            session_root,
            retention_hours=48,
            now=REFERENCE_TIME,
        )

    assert sentinel.read_text() == "keep\n"
    assert session_root.is_symlink()


def test_negative_session_retention_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="non-negative"):
        cleanup_expired_session_storage(tmp_path / "sessions", retention_hours=-1)
