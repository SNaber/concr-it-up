"""Durable SQLite-backed job storage for the KNN-ConcrItUp GUI.

The web process and standalone worker share this Flask-independent store, so
development and hosted deployments use identical queue semantics.
"""

from __future__ import annotations

import copy
import json
import os
import re
import shutil
import sqlite3
import unicodedata
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


ACTIVE_STATUSES = ("preparing", "queued", "running")
TERMINAL_STATUSES = ("done", "error")
ALLOWED_COMMANDS = ("prediction-run", "prediction-holdout")


class JobStoreError(RuntimeError):
    """Base class for durable queue failures."""


class JobAdmissionError(JobStoreError):
    """A job cannot be admitted because a configured limit was reached."""


class JobNotFoundError(JobStoreError):
    """The requested job does not exist."""


@dataclass(frozen=True)
class JobPaths:
    """Filesystem locations reserved for a job."""

    job_id: str
    job_dir: Path
    input_dir: Path
    config_path: Path
    output_dir: Path
    log_path: Path
    manifest_path: Path


@dataclass(frozen=True)
class ClaimedJob:
    """Immutable job data returned by an atomic queue claim."""

    job_id: str
    command: str
    payload: dict[str, Any]
    owner: str
    paths: JobPaths
    timeout_seconds: int
    output_limit_bytes: int
    attempt: int
    lease_owner: str


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""

    return datetime.now(timezone.utc)


def utc_now_iso() -> str:
    """Return an ISO 8601 UTC timestamp."""

    return utc_now().isoformat()


def _parse_iso(value: str | None) -> datetime:
    if not value:
        return datetime.fromtimestamp(0, tz=timezone.utc)
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _json_dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _json_load(value: str | None, default: Any) -> Any:
    if not value:
        return copy.deepcopy(default)
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return copy.deepcopy(default)


def _atomic_json_write(path: Path, payload: Mapping[str, Any]) -> None:
    """Replace a JSON file atomically after writing a sibling temporary file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _safe_input_name(value: str) -> str:
    """Convert a requested bundle input name to one bounded path component."""

    name = Path(str(value)).name
    if not name or name in {".", ".."}:
        raise JobStoreError("Input file name must be a non-empty basename.")
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip(".")
    if not safe:
        raise JobStoreError("Input file name has no safe characters.")
    return safe[:255]


def sanitize_public_message(
    value: object,
    *,
    redact_roots: Iterable[Path | str] = (),
    max_length: int = 1000,
) -> str:
    """Make a log/error line safe and bounded for browser display.

    Server roots and common credential assignments are redacted after Unicode
    normalization and newline folding, preventing private paths or multiline
    output from entering the polling response.
    """

    text = unicodedata.normalize("NFC", str(value)).replace("\x00", "")
    text = " ".join(text.replace("\r", "\n").splitlines()).strip()
    for raw_root in sorted(
        {str(Path(root).resolve(strict=False)) for root in redact_roots},
        key=len,
        reverse=True,
    ):
        if raw_root:
            text = text.replace(raw_root, "<server-path>")
    text = re.sub(r"(?i)(secret|token|password)=([^\s]+)", r"\1=<redacted>", text)
    if len(text) > max_length:
        text = text[: max(0, max_length - 1)] + "…"
    return text


class SQLiteJobStore:
    """SQLite WAL queue with owner scoping and filesystem job bundles.

    Admission and state transitions use short transactions. A worker claims a
    job with ``BEGIN IMMEDIATE`` and may complete or fail it only while holding
    the recorded lease, which preserves the single-running-job invariant
    across separate web and worker processes.
    """

    def __init__(
        self,
        db_path: Path | str,
        job_root: Path | str,
        *,
        global_queue_limit: int = 20,
        per_owner_queue_limit: int = 1,
        storage_stop_fraction: float = 0.80,
        public_log_lines: int = 200,
        redact_roots: Iterable[Path | str] = (),
    ) -> None:
        self.db_path = Path(db_path).resolve(strict=False)
        self.job_root = Path(job_root).resolve(strict=False)
        self.global_queue_limit = int(global_queue_limit)
        self.per_owner_queue_limit = int(per_owner_queue_limit)
        self.storage_stop_fraction = float(storage_stop_fraction)
        self.public_log_lines = max(1, int(public_log_lines))
        self.redact_roots = tuple(Path(item).resolve(strict=False) for item in redact_roots)
        if self.global_queue_limit < 1 or self.per_owner_queue_limit < 1:
            raise ValueError("Queue limits must be positive integers.")
        if not 0.0 < self.storage_stop_fraction <= 1.0:
            raise ValueError("storage_stop_fraction must be in (0, 1].")

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.job_root.mkdir(parents=True, exist_ok=True)
        self._initialize_schema()

    def _connect(self) -> sqlite3.Connection:
        """Open a row-mapped connection with foreign keys and bounded waiting."""

        connection = sqlite3.connect(
            str(self.db_path),
            timeout=30.0,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def _initialize_schema(self) -> None:
        """Create the WAL-backed queue schema and lookup constraints."""

        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=FULL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    command TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    owner TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (
                        status IN ('preparing', 'queued', 'running', 'done', 'error')
                    ),
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT,
                    lease_owner TEXT,
                    lease_expires_at TEXT,
                    attempt INTEGER NOT NULL DEFAULT 0,
                    input_dir TEXT NOT NULL,
                    config_path TEXT NOT NULL,
                    output_dir TEXT NOT NULL,
                    log_path TEXT NOT NULL,
                    manifest_path TEXT NOT NULL,
                    result_json TEXT,
                    public_error TEXT,
                    timeout_seconds INTEGER NOT NULL,
                    output_limit_bytes INTEGER NOT NULL,
                    signature TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_jobs_queue
                    ON jobs(status, created_at, job_id);
                CREATE INDEX IF NOT EXISTS idx_jobs_owner_status
                    ON jobs(owner, status, created_at);
                CREATE INDEX IF NOT EXISTS idx_jobs_finished
                    ON jobs(status, finished_at);
                CREATE UNIQUE INDEX IF NOT EXISTS idx_jobs_owner_signature
                    ON jobs(owner, signature) WHERE signature IS NOT NULL;
                """
            )

    @staticmethod
    def new_job_id() -> str:
        """Return a random, path-safe job identifier."""

        return uuid.uuid4().hex

    def paths_for(self, job_id: str) -> JobPaths:
        """Return canonical bundle paths without creating them."""

        normalized = str(job_id).strip().lower()
        if not re.fullmatch(r"[0-9a-f]{32}", normalized):
            raise JobStoreError("job_id must be a 32-character hexadecimal UUID.")
        job_dir = self.job_root / normalized
        return JobPaths(
            job_id=normalized,
            job_dir=job_dir,
            input_dir=job_dir / "input",
            config_path=job_dir / "config.json",
            output_dir=job_dir / "output",
            log_path=job_dir / "job.log",
            manifest_path=job_dir / "run_manifest.json",
        )

    def storage_usage(self) -> dict[str, float | int]:
        """Return filesystem usage for admission diagnostics."""

        stat = os.statvfs(self.job_root)
        total = int(stat.f_blocks * stat.f_frsize)
        available = int(stat.f_bavail * stat.f_frsize)
        used = max(0, total - available)
        fraction = float(used / total) if total else 1.0
        return {
            "total_bytes": total,
            "available_bytes": available,
            "used_bytes": used,
            "used_fraction": fraction,
        }

    def admission_available(self) -> bool:
        """Return whether job-volume use remains below the stop threshold."""

        return float(self.storage_usage()["used_fraction"]) < self.storage_stop_fraction

    def _reserve_row(
        self,
        *,
        paths: JobPaths,
        command: str,
        payload: Mapping[str, Any],
        owner: str,
        timeout_seconds: int,
        output_limit_bytes: int,
    ) -> None:
        """Reserve a preparing job after atomic global and owner limit checks.

        The immediate transaction serializes competing submissions so neither
        queue limit can be exceeded by concurrent web requests.
        """

        if command not in ALLOWED_COMMANDS:
            raise JobStoreError(f"Unsupported command: {command}")
        if timeout_seconds < 1 or output_limit_bytes < 1:
            raise JobStoreError("Timeout and output limit must be positive.")
        if not self.admission_available():
            percent = round(self.storage_stop_fraction * 100)
            raise JobAdmissionError(
                f"Job storage is at or above the {percent}% admission threshold."
            )

        now = utc_now_iso()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            global_count = int(
                connection.execute(
                    "SELECT COUNT(*) FROM jobs WHERE status IN ('preparing','queued','running')"
                ).fetchone()[0]
            )
            if global_count >= self.global_queue_limit:
                raise JobAdmissionError(
                    f"Global queued/running job limit ({self.global_queue_limit}) reached."
                )
            owner_count = int(
                connection.execute(
                    """SELECT COUNT(*) FROM jobs
                       WHERE owner=? AND status IN ('preparing','queued','running')""",
                    (owner,),
                ).fetchone()[0]
            )
            if owner_count >= self.per_owner_queue_limit:
                raise JobAdmissionError(
                    f"Owner queued/running job limit ({self.per_owner_queue_limit}) reached."
                )
            connection.execute(
                """INSERT INTO jobs (
                       job_id, command, payload_json, owner, status,
                       created_at, updated_at, input_dir, config_path, output_dir,
                       log_path, manifest_path, timeout_seconds, output_limit_bytes
                   ) VALUES (?, ?, ?, ?, 'preparing', ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    paths.job_id,
                    command,
                    _json_dump(dict(payload)),
                    owner,
                    now,
                    now,
                    str(paths.input_dir),
                    str(paths.config_path),
                    str(paths.output_dir),
                    str(paths.log_path),
                    str(paths.manifest_path),
                    timeout_seconds,
                    output_limit_bytes,
                ),
            )
            connection.execute("COMMIT")
        except Exception:
            try:
                connection.execute("ROLLBACK")
            except sqlite3.OperationalError:
                pass
            raise
        finally:
            connection.close()

    def submit(
        self,
        command: str,
        payload: Mapping[str, Any],
        *,
        owner: str,
        config: Mapping[str, Any],
        job_id: str | None = None,
        input_files: Mapping[str, Path | str] | None = None,
        timeout_seconds: int = 7_200,
        output_limit_bytes: int = 250 * 1024 * 1024,
    ) -> str:
        """Create a complete job bundle and transition it to ``queued``.

        A caller-supplied ``job_id`` permits configuration paths to be
        canonicalized before bundle creation. The row remains ``preparing``
        until inputs, config, output directory, and log are all present; bundle
        failures are persisted as terminal errors rather than queued work.
        """

        normalized_owner = str(owner).strip()
        if not normalized_owner:
            raise JobStoreError("owner must be non-empty.")
        paths = self.paths_for(job_id or self.new_job_id())
        self._reserve_row(
            paths=paths,
            command=str(command),
            payload=payload,
            owner=normalized_owner,
            timeout_seconds=int(timeout_seconds),
            output_limit_bytes=int(output_limit_bytes),
        )

        try:
            paths.job_dir.mkdir(parents=False, exist_ok=False)
            paths.input_dir.mkdir()
            paths.output_dir.mkdir()
            for destination_name, source_value in (input_files or {}).items():
                source = Path(source_value).resolve(strict=True)
                if not source.is_file():
                    raise JobStoreError(f"Job input is not a file: {source}")
                destination = paths.input_dir / _safe_input_name(destination_name)
                if destination.exists():
                    raise JobStoreError(f"Duplicate job input name: {destination.name}")
                shutil.copyfile(source, destination)
            _atomic_json_write(paths.config_path, dict(config))
            paths.log_path.touch(exist_ok=False)
        except Exception as exc:
            self.fail(paths.job_id, f"Could not prepare job bundle: {exc}")
            raise

        now = utc_now_iso()
        with self._connect() as connection:
            changed = connection.execute(
                """UPDATE jobs SET status='queued', updated_at=?
                   WHERE job_id=? AND status='preparing'""",
                (now, paths.job_id),
            ).rowcount
        if changed != 1:
            raise JobStoreError("Job preparation state changed unexpectedly.")
        self.append_log(paths.job_id, "Job accepted and queued.")
        return paths.job_id

    def register_existing_result(
        self,
        *,
        command: str,
        payload: Mapping[str, Any],
        result: Mapping[str, Any],
        signature: str,
        owner: str,
    ) -> str:
        """Persist a completed external artifact set once per owner/signature.

        The unique owner/signature index also resolves concurrent discovery
        requests to the same durable record.
        """

        with self._connect() as connection:
            existing = connection.execute(
                "SELECT job_id FROM jobs WHERE owner=? AND signature=?",
                (str(owner), str(signature)),
            ).fetchone()
            if existing is not None:
                return str(existing["job_id"])

        job_id = self.new_job_id()
        paths = self.paths_for(job_id)
        now = utc_now_iso()
        try:
            with self._connect() as connection:
                connection.execute(
                    """INSERT INTO jobs (
                       job_id, command, payload_json, owner, status,
                       created_at, updated_at, started_at, finished_at,
                       input_dir, config_path, output_dir, log_path, manifest_path,
                       result_json, timeout_seconds, output_limit_bytes, signature
                   ) VALUES (?, ?, ?, ?, 'done', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, 1, ?)""",
                    (
                        job_id,
                        str(command),
                        _json_dump(dict(payload)),
                        str(owner),
                        now,
                        now,
                        now,
                        now,
                        str(paths.input_dir),
                        str(paths.config_path),
                        str(paths.output_dir),
                        str(paths.log_path),
                        str(paths.manifest_path),
                        _json_dump(dict(result)),
                        str(signature),
                    ),
                )
        except sqlite3.IntegrityError:
            # A concurrent request may have inserted the same owner/signature
            # after the initial lookup; resolve that race to its durable row.
            with self._connect() as connection:
                existing = connection.execute(
                    "SELECT job_id FROM jobs WHERE owner=? AND signature=?",
                    (str(owner), str(signature)),
                ).fetchone()
            if existing is None:
                raise
            return str(existing["job_id"])
        return job_id

    def count_outstanding(self, *, owner: str | None = None) -> int:
        """Count preparing, queued, and running jobs, optionally by owner."""

        query = "SELECT COUNT(*) FROM jobs WHERE status IN ('preparing','queued','running')"
        params: tuple[Any, ...] = ()
        if owner is not None:
            query += " AND owner=?"
            params = (str(owner),)
        with self._connect() as connection:
            return int(connection.execute(query, params).fetchone()[0])

    def has_active_job(self, *, owner: str | None = None) -> bool:
        """Return whether an owner or the global queue has outstanding work."""

        return self.count_outstanding(owner=owner) > 0

    def active_job(self, *, owner: str | None = None) -> dict[str, Any] | None:
        """Return the newest public outstanding-job record in scope."""

        query = "SELECT * FROM jobs WHERE status IN ('preparing','queued','running')"
        params: tuple[Any, ...] = ()
        if owner is not None:
            query += " AND owner=?"
            params = (str(owner),)
        query += " ORDER BY created_at DESC, job_id DESC LIMIT 1"
        with self._connect() as connection:
            row = connection.execute(query, params).fetchone()
        return self._row_to_public(row) if row is not None else None

    def latest_completed_job(self, *, owner: str | None = None) -> dict[str, Any] | None:
        """Return the most recently updated completed job in scope."""

        query = "SELECT * FROM jobs WHERE status='done'"
        params: tuple[Any, ...] = ()
        if owner is not None:
            query += " AND owner=?"
            params = (str(owner),)
        query += " ORDER BY updated_at DESC, job_id DESC LIMIT 1"
        with self._connect() as connection:
            row = connection.execute(query, params).fetchone()
        return self._row_to_public(row) if row is not None else None

    def get(self, job_id: str, *, owner: str | None = None) -> dict[str, Any] | None:
        """Return a browser-facing job record if it exists in owner scope."""

        query = "SELECT * FROM jobs WHERE job_id=?"
        params: list[Any] = [str(job_id)]
        if owner is not None:
            query += " AND owner=?"
            params.append(str(owner))
        with self._connect() as connection:
            row = connection.execute(query, tuple(params)).fetchone()
        return self._row_to_public(row) if row is not None else None

    def _paths_from_row(self, row: sqlite3.Row) -> JobPaths:
        """Reconstruct trusted bundle paths from an internal database row."""

        return JobPaths(
            job_id=str(row["job_id"]),
            job_dir=Path(str(row["config_path"])).parent,
            input_dir=Path(str(row["input_dir"])),
            config_path=Path(str(row["config_path"])),
            output_dir=Path(str(row["output_dir"])),
            log_path=Path(str(row["log_path"])),
            manifest_path=Path(str(row["manifest_path"])),
        )

    def _public_log(self, log_path: Path) -> list[str]:
        """Read and sanitize a bounded tail of a worker log for polling.

        The byte window prevents polling cost from growing with a long-lived
        log; a truncated first line is discarded before line-count limiting.
        """

        if not log_path.is_file():
            return []
        try:
            max_tail_bytes = 256 * 1024
            with log_path.open("rb") as handle:
                handle.seek(0, os.SEEK_END)
                size = handle.tell()
                offset = max(0, size - max_tail_bytes)
                handle.seek(offset)
                raw_tail = handle.read(max_tail_bytes)
            if offset:
                first_newline = raw_tail.find(b"\n")
                raw_tail = b"" if first_newline < 0 else raw_tail[first_newline + 1 :]
            lines = raw_tail.decode("utf-8", errors="replace").splitlines()
        except OSError:
            return []
        roots = (*self.redact_roots, self.job_root)
        return [
            sanitized
            for raw in lines[-self.public_log_lines :]
            if (sanitized := sanitize_public_message(raw, redact_roots=roots))
        ]

    def _row_to_public(self, row: sqlite3.Row) -> dict[str, Any]:
        """Project a private database row onto the stable JobRecord API shape."""

        paths = self._paths_from_row(row)
        return {
            "job_id": str(row["job_id"]),
            "command": str(row["command"]),
            "payload": _json_load(row["payload_json"], {}),
            "status": str(row["status"]),
            "created_at": str(row["created_at"]),
            "updated_at": str(row["updated_at"]),
            "logs": self._public_log(paths.log_path),
            "result": _json_load(row["result_json"], None),
            "error": row["public_error"],
        }

    def append_log(self, job_id: str, message: object) -> None:
        """Append one worker message and refresh the job update timestamp."""

        with self._connect() as connection:
            row = connection.execute(
                "SELECT log_path FROM jobs WHERE job_id=?", (str(job_id),)
            ).fetchone()
        if row is None:
            raise JobNotFoundError(f"Unknown job id: {job_id}")
        path = Path(str(row["log_path"]))
        path.parent.mkdir(parents=True, exist_ok=True)
        text = str(message).replace("\x00", "")
        with path.open("a", encoding="utf-8") as handle:
            handle.write(text.rstrip("\r\n") + "\n")
        with self._connect() as connection:
            connection.execute(
                "UPDATE jobs SET updated_at=? WHERE job_id=?",
                (utc_now_iso(), str(job_id)),
            )

    def claim_next(self, worker_id: str, *, lease_seconds: int = 60) -> ClaimedJob | None:
        """Atomically lease the oldest queued job when no model job is running.

        The immediate transaction combines the global running-state check,
        FIFO selection, and ``queued`` to ``running`` transition. Concurrent
        workers therefore cannot both obtain model work.
        """

        normalized_worker = str(worker_id).strip()
        if not normalized_worker:
            raise JobStoreError("worker_id must be non-empty.")
        now = utc_now()
        expires = now + timedelta(seconds=max(1, int(lease_seconds)))
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            if int(
                connection.execute("SELECT COUNT(*) FROM jobs WHERE status='running'").fetchone()[0]
            ):
                connection.execute("COMMIT")
                return None
            row = connection.execute(
                """SELECT * FROM jobs WHERE status='queued'
                   ORDER BY created_at ASC, job_id ASC LIMIT 1"""
            ).fetchone()
            if row is None:
                connection.execute("COMMIT")
                return None
            changed = connection.execute(
                """UPDATE jobs
                   SET status='running', started_at=?, updated_at=?, lease_owner=?,
                       lease_expires_at=?, attempt=attempt+1, public_error=NULL
                   WHERE job_id=? AND status='queued'""",
                (
                    now.isoformat(),
                    now.isoformat(),
                    normalized_worker,
                    expires.isoformat(),
                    str(row["job_id"]),
                ),
            ).rowcount
            if changed != 1:
                connection.execute("ROLLBACK")
                return None
            claimed = connection.execute(
                "SELECT * FROM jobs WHERE job_id=?", (str(row["job_id"]),)
            ).fetchone()
            connection.execute("COMMIT")
        except Exception:
            try:
                connection.execute("ROLLBACK")
            except sqlite3.OperationalError:
                pass
            raise
        finally:
            connection.close()

        assert claimed is not None
        return ClaimedJob(
            job_id=str(claimed["job_id"]),
            command=str(claimed["command"]),
            payload=_json_load(claimed["payload_json"], {}),
            owner=str(claimed["owner"]),
            paths=self._paths_from_row(claimed),
            timeout_seconds=int(claimed["timeout_seconds"]),
            output_limit_bytes=int(claimed["output_limit_bytes"]),
            attempt=int(claimed["attempt"]),
            lease_owner=normalized_worker,
        )

    def heartbeat(self, job_id: str, worker_id: str, *, lease_seconds: int = 60) -> bool:
        """Extend a running job lease when it is still owned by ``worker_id``."""

        now = utc_now()
        expires = now + timedelta(seconds=max(1, int(lease_seconds)))
        with self._connect() as connection:
            changed = connection.execute(
                """UPDATE jobs SET updated_at=?, lease_expires_at=?
                   WHERE job_id=? AND status='running' AND lease_owner=?""",
                (now.isoformat(), expires.isoformat(), str(job_id), str(worker_id)),
            ).rowcount
        return changed == 1

    def complete(self, job_id: str, worker_id: str, result: Mapping[str, Any]) -> None:
        """Commit a leased running job as completed with its artifact result."""

        now = utc_now_iso()
        with self._connect() as connection:
            changed = connection.execute(
                """UPDATE jobs
                   SET status='done', result_json=?, public_error=NULL, updated_at=?,
                       finished_at=?, lease_owner=NULL, lease_expires_at=NULL
                   WHERE job_id=? AND status='running' AND lease_owner=?""",
                (_json_dump(dict(result)), now, now, str(job_id), str(worker_id)),
            ).rowcount
        if changed != 1:
            raise JobStoreError("Cannot complete a job without its active lease.")

    def fail(self, job_id: str, error: object, worker_id: str | None = None) -> None:
        """Commit an active job as failed and store a sanitized public error.

        When ``worker_id`` is supplied, the transition succeeds only for that
        lease owner. Preparation failures may omit it because no worker has
        claimed the row yet.
        """

        now = utc_now_iso()
        roots = (*self.redact_roots, self.job_root)
        public_error = sanitize_public_message(error, redact_roots=roots)
        query = """UPDATE jobs
                   SET status='error', public_error=?, updated_at=?, finished_at=?,
                       lease_owner=NULL, lease_expires_at=NULL
                   WHERE job_id=? AND status IN ('preparing','queued','running')"""
        params: list[Any] = [public_error, now, now, str(job_id)]
        if worker_id is not None:
            query += " AND lease_owner=?"
            params.append(str(worker_id))
        with self._connect() as connection:
            changed = connection.execute(query, tuple(params)).rowcount
        if changed != 1:
            raise JobStoreError("Cannot fail a job without its active state/lease.")

    def recover_interrupted_jobs(self, *, reason: str = "Worker restarted during execution.") -> list[str]:
        """Fail jobs interrupted by worker restart while retaining queued work."""

        with self._connect() as connection:
            rows = connection.execute("SELECT job_id FROM jobs WHERE status='running'").fetchall()
        recovered: list[str] = []
        for row in rows:
            job_id = str(row["job_id"])
            self.append_log(job_id, reason)
            self.fail(job_id, reason)
            recovered.append(job_id)
        return recovered

    def cleanup_expired(self, *, retention_hours: float = 48.0, now: datetime | None = None) -> list[str]:
        """Delete expired terminal records and safely contained job bundles.

        A bundle is removed only when its resolved directory is the direct
        child named by its job id beneath ``job_root``. Unexpected database
        paths are skipped rather than passed to recursive deletion.
        """

        if retention_hours < 0:
            raise ValueError("retention_hours must be non-negative.")
        cutoff = (now or utc_now()) - timedelta(hours=float(retention_hours))
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT job_id, config_path, finished_at, updated_at FROM jobs
                   WHERE status IN ('done','error')"""
            ).fetchall()

        removed: list[str] = []
        root = self.job_root.resolve(strict=False)
        for row in rows:
            timestamp = _parse_iso(row["finished_at"] or row["updated_at"])
            if timestamp > cutoff:
                continue
            job_id = str(row["job_id"])
            job_dir = Path(str(row["config_path"])).parent.resolve(strict=False)
            if job_dir.parent != root or job_dir.name != job_id:
                continue
            if job_dir.exists():
                shutil.rmtree(job_dir)
            with self._connect() as connection:
                connection.execute("DELETE FROM jobs WHERE job_id=?", (job_id,))
            removed.append(job_id)
        return removed

    def journal_mode(self) -> str:
        """Return SQLite's active journal mode for runtime diagnostics."""

        with self._connect() as connection:
            return str(connection.execute("PRAGMA journal_mode").fetchone()[0]).lower()
