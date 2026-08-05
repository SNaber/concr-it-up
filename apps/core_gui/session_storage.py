"""Activity tracking and bounded retention for hosted owner storage."""

from __future__ import annotations

try:
    import fcntl
except ImportError:  # pragma: no cover - hosted deployment is POSIX-only
    fcntl = None  # type: ignore[assignment]
import logging
import os
import re
import shutil
import stat
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import BinaryIO, Callable, Iterable


ACTIVITY_FILENAME = ".last_activity"
LOCK_DIRECTORY = ".locks"
TRASH_DIRECTORY = ".trash"
OWNER_STORAGE_PATTERN = re.compile(r"[0-9a-f]{64}")
TOMBSTONE_PATTERN = re.compile(r"[0-9a-f]{64}\.[0-9a-f]{32}")

LOGGER = logging.getLogger(__name__)


class SessionStorageError(RuntimeError):
    """Hosted owner storage cannot be accessed safely."""


class SessionStorageLease:
    """A process-wide shared or exclusive lock held through an open file."""

    def __init__(self, handle: BinaryIO) -> None:
        self._handle = handle

    @property
    def closed(self) -> bool:
        return self._handle.closed

    def close(self) -> None:
        if self._handle.closed:
            return
        try:
            _require_posix_locks()
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        finally:
            self._handle.close()

    def __enter__(self) -> "SessionStorageLease":
        return self

    def __exit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        self.close()


def _require_storage_key(storage_key: str) -> str:
    normalized = str(storage_key).strip().lower()
    if not OWNER_STORAGE_PATTERN.fullmatch(normalized):
        raise SessionStorageError("Owner storage key must be 64 lowercase hexadecimal characters.")
    return normalized


def _require_posix_locks() -> None:
    if fcntl is None:
        raise SessionStorageError("Hosted session retention requires POSIX file locks.")


def _lexical_absolute(path: Path | str) -> Path:
    """Return an absolute path without following its final symlink."""

    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    return candidate.absolute()


def validate_session_root(
    session_root: Path | str,
    *,
    managed_data_root: Path | str,
    disallowed_paths: Iterable[Path | str] = (),
) -> Path:
    """Validate the administrator-configured deletion root.

    Session cleanup is intentionally limited to a dedicated directory strictly
    below the repository's managed ``data`` directory. The configured final
    component may not be a symlink, and it may not contain or be contained by
    the job root or queue database path.
    """

    lexical = _lexical_absolute(session_root)
    try:
        metadata = lexical.lstat()
    except FileNotFoundError:
        metadata = None
    if metadata is not None and stat.S_ISLNK(metadata.st_mode):
        raise SessionStorageError(f"Managed session root may not be a symlink: {lexical}")

    resolved = lexical.resolve(strict=False)
    data_root = Path(managed_data_root).expanduser().resolve(strict=False)
    try:
        relative = resolved.relative_to(data_root)
    except ValueError as exc:
        raise SessionStorageError(
            f"Managed session root must be below the repository data directory: {data_root}"
        ) from exc
    if not relative.parts:
        raise SessionStorageError(
            "Managed session root must be a dedicated child of the repository data directory."
        )

    for raw_disallowed in disallowed_paths:
        disallowed = Path(raw_disallowed).expanduser().resolve(strict=False)
        overlaps = (
            resolved == disallowed
            or resolved in disallowed.parents
            or disallowed in resolved.parents
        )
        if overlaps:
            raise SessionStorageError(
                f"Managed session root overlaps another managed path: {disallowed}"
            )
    return resolved


def _ensure_real_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    metadata = path.lstat()
    if path.is_symlink() or not stat.S_ISDIR(metadata.st_mode):
        raise SessionStorageError(f"Managed session path is not a real directory: {path}")


def _open_regular_file(path: Path) -> BinaryIO:
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as exc:
        raise SessionStorageError(f"Could not open managed session metadata: {path}") from exc
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode):
        os.close(descriptor)
        raise SessionStorageError(f"Managed session metadata is not a regular file: {path}")
    return os.fdopen(descriptor, "r+b", buffering=0)


def _touch_activity(path: Path, *, timestamp: float | None = None) -> None:
    handle = _open_regular_file(path)
    try:
        if timestamp is None:
            os.utime(handle.fileno(), None)
        else:
            os.utime(handle.fileno(), (timestamp, timestamp))
    finally:
        handle.close()


def _lock_path(session_root: Path, storage_key: str) -> Path:
    lock_root = session_root / LOCK_DIRECTORY
    _ensure_real_directory(lock_root)
    return lock_root / f"{storage_key[:2]}.lock"


def _activate_owner_storage(session_root: Path, storage_key: str) -> Path:
    """Create one owner tree and refresh it while its stripe lock is held."""

    owner_root = session_root / storage_key
    _ensure_real_directory(owner_root)
    _touch_activity(owner_root / ACTIVITY_FILENAME)
    return owner_root


def activate_session_storage(session_root: Path | str, storage_key: str) -> Path:
    """Activate owner storage while the caller holds its shared stripe lock."""

    root = _lexical_absolute(session_root)
    key = _require_storage_key(storage_key)
    _ensure_real_directory(root)
    return _activate_owner_storage(root, key)


def acquire_session_storage_lease(
    session_root: Path | str,
    storage_key: str,
    *,
    create_owner: bool = True,
) -> SessionStorageLease:
    """Lock one owner stripe and optionally activate its persistent tree.

    With ``create_owner=False``, an existing owner tree is refreshed but a new
    browser cookie does not create a persistent directory merely by issuing a
    read-only request.
    """

    _require_posix_locks()
    root = _lexical_absolute(session_root)
    key = _require_storage_key(storage_key)
    _ensure_real_directory(root)
    handle = _open_regular_file(_lock_path(root, key))
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
        owner_root = root / key
        if _is_real_directory(owner_root):
            _touch_activity(owner_root / ACTIVITY_FILENAME)
        else:
            try:
                owner_root.lstat()
            except FileNotFoundError:
                if create_owner:
                    _activate_owner_storage(root, key)
            else:
                raise SessionStorageError(
                    f"Managed owner path is not a real directory: {owner_root}"
                )
    except Exception:
        handle.close()
        raise
    return SessionStorageLease(handle)


def _try_exclusive_lease(session_root: Path, storage_key: str) -> SessionStorageLease | None:
    _require_posix_locks()
    handle = _open_regular_file(_lock_path(session_root, storage_key))
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return None
    except Exception:
        handle.close()
        raise
    return SessionStorageLease(handle)


def _is_real_directory(path: Path) -> bool:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return False
    return not path.is_symlink() and stat.S_ISDIR(metadata.st_mode)


def _activity_timestamp(path: Path) -> float | None:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return None
    if path.is_symlink() or not stat.S_ISREG(metadata.st_mode):
        raise SessionStorageError(f"Invalid session activity marker: {path}")
    return float(metadata.st_mtime)


def _remove_tombstone(path: Path) -> bool:
    if not TOMBSTONE_PATTERN.fullmatch(path.name) or not _is_real_directory(path):
        return False
    try:
        shutil.rmtree(path)
    except OSError as exc:
        LOGGER.warning("Could not remove expired session tombstone %s: %s", path.name, exc)
        return False
    return True


def cleanup_expired_session_storage(
    session_root: Path | str,
    *,
    active_storage_keys: Iterable[str] = (),
    active_storage_keys_provider: Callable[[], Iterable[str]] | None = None,
    retention_hours: float = 48.0,
    now: datetime | None = None,
) -> list[str]:
    """Delete inactive owner trees after an exclusive-lock safety recheck.

    Only real, direct children named by a 64-hex owner storage key are
    considered. Existing pre-retention directories without an activity marker
    receive a full grace period. Expired trees are renamed into a private trash
    directory before recursive removal, so a returning browser can immediately
    create a new empty owner tree after the exclusive lock is released.
    """

    if retention_hours < 0:
        raise ValueError("retention_hours must be non-negative.")
    reference = now or datetime.now(timezone.utc)
    if reference.tzinfo is None:
        reference = reference.replace(tzinfo=timezone.utc)
    reference_timestamp = reference.timestamp()
    cutoff_timestamp = (reference - timedelta(hours=float(retention_hours))).timestamp()

    _require_posix_locks()
    root = _lexical_absolute(session_root)
    _ensure_real_directory(root)
    trash_root = root / TRASH_DIRECTORY
    _ensure_real_directory(trash_root)

    for tombstone in sorted(trash_root.iterdir(), key=lambda item: item.name):
        _remove_tombstone(tombstone)

    active = {
        normalized
        for value in active_storage_keys
        if OWNER_STORAGE_PATTERN.fullmatch(normalized := str(value).strip().lower())
    }
    removed: list[str] = []
    for candidate in sorted(root.iterdir(), key=lambda item: item.name):
        key = candidate.name
        if not OWNER_STORAGE_PATTERN.fullmatch(key) or key in active:
            continue
        if not _is_real_directory(candidate):
            continue

        try:
            lease = _try_exclusive_lease(root, key)
        except SessionStorageError as exc:
            LOGGER.warning("Could not lock hosted session %s: %s", key, exc)
            continue
        if lease is None:
            continue

        tombstone: Path | None = None
        try:
            if key in active or not _is_real_directory(candidate):
                continue
            marker = candidate / ACTIVITY_FILENAME
            try:
                activity_timestamp = _activity_timestamp(marker)
            except SessionStorageError as exc:
                LOGGER.warning("Skipping hosted session %s: %s", key, exc)
                continue
            if activity_timestamp is None:
                _touch_activity(marker, timestamp=reference_timestamp)
                continue
            if activity_timestamp > cutoff_timestamp:
                continue

            # This lookup deliberately happens while the owner's exclusive
            # stripe lock is held. Web submissions hold the shared form of the
            # same lock, so an owner cannot become queued between this final
            # database check and quarantine.
            if active_storage_keys_provider is not None:
                try:
                    current_active = {
                        normalized
                        for value in active_storage_keys_provider()
                        if OWNER_STORAGE_PATTERN.fullmatch(
                            normalized := str(value).strip().lower()
                        )
                    }
                except Exception as exc:
                    LOGGER.warning(
                        "Could not recheck active hosted sessions before deleting %s: %s",
                        key,
                        exc,
                    )
                    continue
                if key in current_active:
                    continue

            tombstone = trash_root / f"{key}.{uuid.uuid4().hex}"
            os.replace(candidate, tombstone)
        except OSError as exc:
            LOGGER.warning("Could not quarantine expired hosted session %s: %s", key, exc)
            continue
        finally:
            lease.close()

        if tombstone is not None:
            _remove_tombstone(tombstone)
            removed.append(key)
    return removed
