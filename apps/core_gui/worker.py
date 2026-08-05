"""Single-process durable worker for hosted KNN-ConcrItUp jobs.

The worker atomically claims one SQLite job and starts one fresh core CLI
subprocess. Production deployments run it separately from Gunicorn, and
embedding models are loaded only inside the child process.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import signal
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .hosted_config import owner_storage_key
from .job_store import ClaimedJob, JobStoreError, SQLiteJobStore, sanitize_public_message
from .session_storage import cleanup_expired_session_storage, validate_session_root


ARTIFACT_FILENAMES: dict[str, str] = {
    "summary": "summary.json",
    "cv_results": "cv_results.csv",
    "test_predictions": "test_predictions.csv",
    "vocab_predictions": "vocab_predictions.csv",
    "oov_gold": "oov_gold.csv",
    "oov_vocab": "oov_vocab.csv",
    "holdout_summary": "holdout_summary.json",
    "holdout_unmatched": "holdout_unmatched.csv",
}

MANIFEST_PACKAGES = (
    "concreteness-knn-core",
    "Flask",
    "fasttext",
    "gensim",
    "joblib",
    "numpy",
    "pandas",
    "scikit-learn",
    "scipy",
)


def _env_absolute_path(name: str, default: Path, *, repo_root: Path) -> Path:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default.absolute()
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    return candidate.absolute()


def _env_path(name: str, default: Path, *, repo_root: Path) -> Path:
    return _env_absolute_path(name, default, repo_root=repo_root).resolve(strict=False)


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    raw = os.environ.get(name, "").strip()
    value = default if not raw else int(raw)
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}.")
    return value


def _env_float(name: str, default: float, *, minimum: float = 0.0) -> float:
    raw = os.environ.get(name, "").strip()
    value = default if not raw else float(raw)
    if not math.isfinite(value) or value < minimum:
        raise ValueError(f"{name} must be a finite number >= {minimum}.")
    return value


@dataclass(frozen=True)
class WorkerSettings:
    """Resolved paths, timing, retention, and queue limits for one worker."""

    repo_root: Path
    db_path: Path
    job_root: Path
    entrypoint: str
    worker_id: str
    session_root: Path | None = None
    poll_seconds: float = 1.0
    lease_seconds: int = 60
    retention_hours: float = 48.0
    cleanup_interval_seconds: int = 300
    global_queue_limit: int = 20
    per_owner_queue_limit: int = 1
    storage_stop_fraction: float = 0.80

    @classmethod
    def from_env(cls) -> "WorkerSettings":
        """Resolve worker settings from ``CONCRITUP_*`` environment values."""

        module_repo = Path(__file__).resolve().parents[2]
        repo_root = _env_path("CONCRITUP_REPO_ROOT", module_repo, repo_root=module_repo)
        job_root = _env_path(
            "CONCRITUP_JOB_ROOT",
            repo_root / "data" / "jobs",
            repo_root=repo_root,
        )
        db_path = _env_path(
            "CONCRITUP_JOB_DB",
            repo_root / "data" / "job_queue.sqlite3",
            repo_root=repo_root,
        )
        session_candidate = _env_absolute_path(
            "CONCRITUP_SESSION_ROOT",
            repo_root / "data" / "hosted_sessions",
            repo_root=repo_root,
        )
        session_root = validate_session_root(
            session_candidate,
            managed_data_root=repo_root / "data",
            disallowed_paths=(job_root, db_path),
        )
        default_entrypoint = repo_root / ".venv" / "bin" / "concreteness-knn-core"
        entrypoint = os.environ.get("CONCRITUP_CORE_ENTRYPOINT", "").strip()
        if not entrypoint:
            entrypoint = str(default_entrypoint if default_entrypoint.exists() else "concreteness-knn-core")
        worker_id = os.environ.get("CONCRITUP_WORKER_ID", "").strip()
        if not worker_id:
            worker_id = f"{socket.gethostname()}-{os.getpid()}"
        storage_percent = _env_float("CONCRITUP_STORAGE_STOP_PERCENT", 80.0)
        if not 0.0 < storage_percent <= 100.0:
            raise ValueError("CONCRITUP_STORAGE_STOP_PERCENT must be in (0, 100].")
        return cls(
            repo_root=repo_root,
            db_path=db_path,
            job_root=job_root,
            entrypoint=entrypoint,
            worker_id=worker_id,
            session_root=session_root,
            poll_seconds=_env_float("CONCRITUP_WORKER_POLL_SECONDS", 1.0, minimum=0.05),
            lease_seconds=_env_int("CONCRITUP_WORKER_LEASE_SECONDS", 60),
            retention_hours=_env_float("CONCRITUP_RETENTION_HOURS", 48.0),
            cleanup_interval_seconds=_env_int("CONCRITUP_CLEANUP_INTERVAL_SECONDS", 300),
            global_queue_limit=_env_int("CONCRITUP_GLOBAL_JOB_LIMIT", 20),
            per_owner_queue_limit=_env_int("CONCRITUP_OWNER_JOB_LIMIT", 1),
            storage_stop_fraction=storage_percent / 100.0,
        )


def create_store(settings: WorkerSettings) -> SQLiteJobStore:
    """Create the durable store configured for a worker process."""

    return SQLiteJobStore(
        settings.db_path,
        settings.job_root,
        global_queue_limit=settings.global_queue_limit,
        per_owner_queue_limit=settings.per_owner_queue_limit,
        storage_stop_fraction=settings.storage_stop_fraction,
        redact_roots=(settings.repo_root,),
    )


def sha256_file(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    """Hash a file with bounded memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def directory_size(path: Path) -> int:
    """Return the bytes occupied by regular files below ``path``."""

    total = 0
    if not path.exists():
        return 0
    for candidate in path.rglob("*"):
        try:
            if candidate.is_file() and not candidate.is_symlink():
                total += candidate.stat().st_size
        except FileNotFoundError:
            continue
    return total


def _resolve_runtime_path(value: object, *, repo_root: Path) -> Path | None:
    """Resolve an optional config path against the worker repository root."""

    raw = str(value or "").strip()
    if not raw:
        return None
    path = Path(raw)
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve(strict=False)


def _active_embedding_configs(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return only embedding entries active for the configured mode."""

    embeddings = config.get("embeddings")
    if not isinstance(embeddings, Mapping):
        return []
    spaces = [dict(item) for item in embeddings.get("spaces", []) if isinstance(item, Mapping)]
    mode = str(embeddings.get("mode", "single"))
    if mode != "single":
        return spaces
    active = str(embeddings.get("active_space", ""))
    return [item for item in spaces if str(item.get("id", "")) == active]


def _checksum_entry(path: Path, *, logical_path: str) -> dict[str, Any]:
    """Describe file existence, size, and digest without exposing its path."""

    if not path.is_file():
        return {
            "logical_path": logical_path,
            "exists": False,
            "sha256": None,
            "size_bytes": None,
        }
    stat = path.stat()
    return {
        "logical_path": logical_path,
        "exists": True,
        "sha256": sha256_file(path),
        "size_bytes": int(stat.st_size),
    }


def collect_checksums(
    config: Mapping[str, Any], *, repo_root: Path, command: str = "prediction-run"
) -> dict[str, Any]:
    """Checksum every scientific input read by the selected core command.

    Holdout scoring records only its holdout and prediction tables; model runs
    additionally record gold, target, and active embedding inputs.
    """

    dataset = config.get("dataset") if isinstance(config.get("dataset"), Mapping) else {}
    prediction = config.get("prediction") if isinstance(config.get("prediction"), Mapping) else {}
    if command == "prediction-holdout":
        input_values = {
            "holdout": prediction.get("holdout"),
            "predictions": prediction.get("predictions_csv"),
        }
    else:
        input_values = {
            "gold": dataset.get("gold"),
            "target": prediction.get("target"),
        }
    inputs: dict[str, Any] = {}
    for label, value in input_values.items():
        path = _resolve_runtime_path(value, repo_root=repo_root)
        if path is not None:
            inputs[label] = _checksum_entry(path, logical_path=f"input:{label}/{path.name}")

    embeddings: dict[str, Any] = {}
    active_spaces = [] if command == "prediction-holdout" else _active_embedding_configs(config)
    for space in active_spaces:
        space_id = str(space.get("id", ""))
        path = _resolve_runtime_path(space.get("path"), repo_root=repo_root)
        if space_id and path is not None:
            embeddings[space_id] = {
                **_checksum_entry(
                    path,
                    logical_path=f"embedding:{space_id}/{path.name}",
                ),
                "kind": str(space.get("kind", "")),
                "label": str(space.get("label", "")),
            }
    return {"inputs": inputs, "embeddings": embeddings}


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in MANIFEST_PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def _git_revision(repo_root: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return "unavailable"
    return completed.stdout.strip() or "unavailable"


def _load_json_if_present(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _logical_server_path(value: str, *, job: ClaimedJob, settings: WorkerSettings) -> str:
    """Replace a resolved server path with a stable, non-sensitive alias."""

    candidate = Path(value)
    if not candidate.is_absolute():
        return value
    resolved = candidate.resolve(strict=False)
    roots = (
        (job.paths.job_dir.resolve(strict=False), "job"),
        (settings.repo_root.resolve(strict=False), "repository"),
    )
    for root, prefix in roots:
        try:
            relative = resolved.relative_to(root)
        except ValueError:
            continue
        return f"{prefix}/{relative.as_posix()}"
    return f"server-file/{resolved.name}"


def _redact_manifest_value(value: Any, *, job: ClaimedJob, settings: WorkerSettings) -> Any:
    """Recursively replace absolute paths in manifest-bound values."""

    if isinstance(value, Mapping):
        return {
            str(key): _redact_manifest_value(item, job=job, settings=settings)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_manifest_value(item, job=job, settings=settings) for item in value]
    if isinstance(value, tuple):
        return [_redact_manifest_value(item, job=job, settings=settings) for item in value]
    if isinstance(value, str):
        stripped = value.strip()
        if Path(stripped).is_absolute():
            return _logical_server_path(stripped, job=job, settings=settings)
        for root in (settings.repo_root, settings.job_root):
            raw_root = str(root.resolve(strict=False))
            if raw_root in value:
                value = value.replace(raw_root, "<server-path>")
        return value
    return value


def _manifest_config(
    config: Mapping[str, Any], *, job: ClaimedJob, settings: WorkerSettings
) -> dict[str, Any]:
    """Create a public resolved config with logical embedding/path aliases."""

    public_config = _redact_manifest_value(dict(config), job=job, settings=settings)
    embeddings = public_config.get("embeddings")
    if isinstance(embeddings, dict):
        for space in embeddings.get("spaces", []):
            if isinstance(space, dict):
                space["path"] = f"embedding:{space.get('id', 'selected')}"
    return public_config


def discover_artifacts(job: ClaimedJob) -> dict[str, str]:
    """Discover core artifacts and require the command's essential outputs."""

    artifacts: dict[str, str] = {}
    for artifact_key, filename in ARTIFACT_FILENAMES.items():
        candidate = job.paths.output_dir / filename
        if candidate.is_file():
            artifacts[artifact_key] = str(candidate.resolve())
    if job.command == "prediction-run":
        missing = [key for key in ("summary", "cv_results", "test_predictions") if key not in artifacts]
    else:
        missing = [key for key in ("holdout_summary",) if key not in artifacts]
    if missing:
        raise JobStoreError(f"Core command completed without required artifacts: {', '.join(missing)}")
    return artifacts


def build_run_manifest(
    job: ClaimedJob,
    *,
    settings: WorkerSettings,
    config: Mapping[str, Any],
    command_argv: Sequence[str],
    started_at: str,
    finished_at: str,
    elapsed_seconds: float,
    return_code: int | None,
    status: str,
    artifacts: Mapping[str, str],
    error: str | None,
) -> dict[str, Any]:
    """Build a path-safe provenance manifest alongside core artifacts.

    The manifest records scientific input checksums, resolved configuration,
    software versions, timing, and summary metadata. Owner identities and
    absolute server paths are excluded from the public record.
    """

    summary = _load_json_if_present(Path(artifacts["summary"])) if "summary" in artifacts else None
    holdout = (
        _load_json_if_present(Path(artifacts["holdout_summary"]))
        if "holdout_summary" in artifacts
        else None
    )
    dataset = config.get("dataset") if isinstance(config.get("dataset"), Mapping) else {}
    return {
        "manifest_version": "1.0.0",
        "job": {
            "job_id": job.job_id,
            "command": job.command,
            "attempt": job.attempt,
            "status": status,
            "started_at": started_at,
            "finished_at": finished_at,
            "elapsed_seconds": float(elapsed_seconds),
            "return_code": return_code,
            "timeout_seconds": job.timeout_seconds,
            "output_limit_bytes": job.output_limit_bytes,
            "error": error,
        },
        "command_argv": [
            Path(str(command_argv[0])).name,
            *[str(item) for item in command_argv[1:-1]],
            "job/config.json",
        ],
        "resolved_config": _manifest_config(config, job=job, settings=settings),
        "checksums": collect_checksums(
            config, repo_root=settings.repo_root, command=job.command
        ),
        "software": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "git_revision": _git_revision(settings.repo_root),
            "packages": _package_versions(),
        },
        "normalization": {
            "unicode": "No Unicode normalization; core uses strip plus optional lowercase.",
            "lowercase": bool(dataset.get("lowercase", True)),
            "pos_filter": dataset.get("pos_filter", {}),
        },
        "evaluation_and_coverage": {
            "prediction_summary": _redact_manifest_value(
                summary, job=job, settings=settings
            ),
            "holdout_summary": _redact_manifest_value(
                holdout, job=job, settings=settings
            ),
        },
        "artifacts": {
            key: _logical_server_path(value, job=job, settings=settings)
            for key, value in artifacts.items()
        },
    }


def _atomic_manifest_write(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically replace a run manifest through a sibling temporary file."""

    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


Executor = Callable[[ClaimedJob, Sequence[str], Path, Mapping[str, str]], int]


class JobWorker:
    """Run leased jobs sequentially in fresh core CLI subprocesses.

    Queue ownership remains in :class:`SQLiteJobStore`; this class supplies
    lease heartbeats, runtime/output enforcement, log capture, provenance, and
    terminal state transitions. Callers may supply an alternate executor.
    """

    def __init__(
        self,
        store: SQLiteJobStore,
        settings: WorkerSettings,
        *,
        executor: Executor | None = None,
    ) -> None:
        self.store = store
        self.settings = settings
        self.executor = executor
        self.stop_event = threading.Event()
        self._last_cleanup: float | None = None

    def command_for(self, job: ClaimedJob) -> list[str]:
        """Build the shell-free core CLI argument vector for a claimed job."""

        subcommand = "run" if job.command == "prediction-run" else "holdout"
        return [
            self.settings.entrypoint,
            "prediction",
            subcommand,
            "--config",
            str(job.paths.config_path),
        ]

    @staticmethod
    def subprocess_environment() -> dict[str, str]:
        """Return an environment with bounded numerical-library thread counts."""

        environment = dict(os.environ)
        environment["SKLEARN_WORKING_MEMORY"] = environment.get("SKLEARN_WORKING_MEMORY", "256")
        for name in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "BLIS_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
        ):
            environment[name] = "1"
        return environment

    def _execute_subprocess(
        self,
        job: ClaimedJob,
        command: Sequence[str],
        cwd: Path,
        environment: Mapping[str, str],
    ) -> int:
        """Run the core command while enforcing lease, timeout, and output caps.

        Combined stdout and stderr are appended directly to the private job
        log. Any enforcement or lease failure terminates the child, escalating
        to a kill if it does not exit promptly.
        """

        started = time.monotonic()
        heartbeat_at = started
        with job.paths.log_path.open("ab", buffering=0) as log_handle:
            process = subprocess.Popen(
                list(command),
                cwd=str(cwd),
                env=dict(environment),
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                shell=False,
            )
            try:
                while process.poll() is None:
                    now = time.monotonic()
                    if now - started > job.timeout_seconds:
                        raise TimeoutError(
                            f"Model subprocess exceeded the {job.timeout_seconds}-second timeout."
                        )
                    if directory_size(job.paths.output_dir) > job.output_limit_bytes:
                        raise JobStoreError(
                            f"Model output exceeded the {job.output_limit_bytes}-byte limit."
                        )
                    if now - heartbeat_at >= max(1.0, self.settings.lease_seconds / 3):
                        if not self.store.heartbeat(
                            job.job_id,
                            self.settings.worker_id,
                            lease_seconds=self.settings.lease_seconds,
                        ):
                            raise JobStoreError("Worker lost its job lease.")
                        heartbeat_at = now
                    time.sleep(min(0.5, self.settings.poll_seconds))
            except Exception:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
                raise
            return int(process.wait())

    def _write_failure_manifest(
        self,
        job: ClaimedJob,
        *,
        config: Mapping[str, Any],
        command: Sequence[str],
        started_at: str,
        started_monotonic: float,
        return_code: int | None,
        error: str,
    ) -> None:
        """Best-effort provenance writing for a failed model attempt.

        Manifest failures are logged but do not replace the original public
        job error or prevent the durable failure transition.
        """

        try:
            manifest = build_run_manifest(
                job,
                settings=self.settings,
                config=config,
                command_argv=command,
                started_at=started_at,
                finished_at=datetime.now(timezone.utc).isoformat(),
                elapsed_seconds=time.monotonic() - started_monotonic,
                return_code=return_code,
                status="error",
                artifacts={},
                error=error,
            )
            _atomic_manifest_write(job.paths.manifest_path, manifest)
        except Exception as manifest_exc:
            self.store.append_log(job.job_id, f"Could not write failure manifest: {manifest_exc}")

    def execute(self, job: ClaimedJob) -> None:
        """Execute a claimed job and commit exactly one terminal outcome.

        Successful commands must produce their required core artifacts before
        the manifest and ``done`` transition are written. Exceptions are
        sanitized, logged, represented in a failure manifest when possible,
        and committed through the active lease.
        """

        started_at = datetime.now(timezone.utc).isoformat()
        started_monotonic = time.monotonic()
        command = self.command_for(job)
        return_code: int | None = None
        config: dict[str, Any] = {}
        try:
            loaded = json.loads(job.paths.config_path.read_text(encoding="utf-8"))
            if not isinstance(loaded, dict):
                raise JobStoreError("Job config must contain a JSON object.")
            config = loaded
            self.store.append_log(job.job_id, f"Starting {job.command} model subprocess.")
            executor = self.executor or self._execute_subprocess
            return_code = int(
                executor(job, command, self.settings.repo_root, self.subprocess_environment())
            )
            if return_code != 0:
                raise JobStoreError(f"Model subprocess exited with status {return_code}.")
            if directory_size(job.paths.output_dir) > job.output_limit_bytes:
                raise JobStoreError(
                    f"Model output exceeded the {job.output_limit_bytes}-byte limit."
                )
            artifacts = discover_artifacts(job)
            finished_at = datetime.now(timezone.utc).isoformat()
            manifest = build_run_manifest(
                job,
                settings=self.settings,
                config=config,
                command_argv=command,
                started_at=started_at,
                finished_at=finished_at,
                elapsed_seconds=time.monotonic() - started_monotonic,
                return_code=return_code,
                status="done",
                artifacts=artifacts,
                error=None,
            )
            _atomic_manifest_write(job.paths.manifest_path, manifest)
            artifacts = {**artifacts, "run_manifest": str(job.paths.manifest_path.resolve())}
            self.store.append_log(job.job_id, "Model subprocess completed successfully.")
            self.store.complete(
                job.job_id,
                self.settings.worker_id,
                {"config_path": str(job.paths.config_path), "artifacts": artifacts},
            )
        except Exception as exc:
            public_error = sanitize_public_message(
                exc,
                redact_roots=(self.settings.repo_root, self.settings.job_root),
            )
            try:
                self.store.append_log(job.job_id, f"Job failed: {public_error}")
            except JobStoreError:
                pass
            self._write_failure_manifest(
                job,
                config=config,
                command=command,
                started_at=started_at,
                started_monotonic=started_monotonic,
                return_code=return_code,
                error=public_error,
            )
            self.store.fail(job.job_id, public_error, worker_id=self.settings.worker_id)

    def run_once(self) -> bool:
        """Claim and execute at most one job, reporting whether work existed."""

        job = self.store.claim_next(
            self.settings.worker_id,
            lease_seconds=self.settings.lease_seconds,
        )
        if job is None:
            return False
        self.execute(job)
        return True

    def recover(self) -> list[str]:
        """Mark jobs left running by an earlier worker process as failed."""

        return self.store.recover_interrupted_jobs()

    def cleanup_if_due(self, *, force: bool = False) -> list[str]:
        """Run job and owner-session retention when the interval has elapsed."""

        now = time.monotonic()
        if (
            not force
            and self._last_cleanup is not None
            and now - self._last_cleanup < self.settings.cleanup_interval_seconds
        ):
            return []
        removed_jobs = self.store.cleanup_expired(
            retention_hours=self.settings.retention_hours
        )
        removed_sessions: list[str] = []
        if self.settings.session_root is not None:
            def active_storage_keys() -> set[str]:
                return {owner_storage_key(owner) for owner in self.store.active_owners()}

            removed_sessions = cleanup_expired_session_storage(
                self.settings.session_root,
                active_storage_keys=active_storage_keys(),
                active_storage_keys_provider=active_storage_keys,
                retention_hours=self.settings.retention_hours,
            )
        self._last_cleanup = now
        return [*removed_jobs, *(f"session:{key}" for key in removed_sessions)]

    def run_forever(self, *, recover: bool = True) -> None:
        """Poll, clean, and execute jobs until cooperative shutdown is requested."""

        if recover:
            self.recover()
        while not self.stop_event.is_set():
            self.cleanup_if_due()
            if not self.run_once():
                self.stop_event.wait(self.settings.poll_seconds)

    def stop(self) -> None:
        """Request cooperative termination of the worker polling loop."""

        self.stop_event.set()


def build_parser() -> argparse.ArgumentParser:
    """Build the standalone worker command-line parser."""

    parser = argparse.ArgumentParser(description="Run the durable KNN-ConcrItUp worker.")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Claim at most one queued job and exit.",
    )
    parser.add_argument(
        "--no-recover",
        action="store_true",
        help="Do not mark jobs left running by an earlier worker as interrupted.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one worker process and return its command-line exit status."""

    args = build_parser().parse_args(argv)
    settings = WorkerSettings.from_env()
    store = create_store(settings)
    worker = JobWorker(store, settings)

    def request_stop(_signum: int, _frame: object) -> None:
        worker.stop()

    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGTERM, request_stop)
        signal.signal(signal.SIGINT, request_stop)

    if not args.no_recover:
        worker.recover()
    if args.once:
        worker.cleanup_if_due(force=True)
        worker.run_once()
        return 0
    worker.run_forever(recover=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
