"""Flask GUI for core configuration editing and workflow execution."""

from __future__ import annotations

import copy
import contextlib
import csv
import hashlib
import hmac
import json
import os
import re
import secrets
import sys
import threading
from pathlib import Path
from typing import Any, Callable

from flask import Flask, Response, g, jsonify, render_template, request, send_file, session
from werkzeug.exceptions import RequestEntityTooLarge

DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[2]
CORE_SRC = DEFAULT_REPO_ROOT / "packages" / "core" / "src"
if str(CORE_SRC) not in sys.path:
    sys.path.insert(0, str(CORE_SRC))

from concreteness_knn_core.config import (  # noqa: E402
    default_config as core_default_config,
    load_config as core_load_config,
    validate_config as core_validate_config,
)
from concreteness_knn_core.prediction import PredictionPipeline  # noqa: E402

from .hosted_config import (  # noqa: E402
    HostedConfigError,
    HostedPathResolver,
    HostedPolicy,
    HostedSettings,
)
from .job_store import (  # noqa: E402
    JobAdmissionError,
    JobPaths,
    SQLiteJobStore,
    sanitize_public_message,
)
from .worker import JobWorker, WorkerSettings  # noqa: E402

MAX_RESULT_PREVIEW_LIMIT = 200
DEFAULT_RESULT_PREVIEW_LIMIT = 50
WORD_SEARCH_COLUMN_CANDIDATES = (
    "word",
    "_word",
    "target_word",
    "target",
    "token",
    "lemma",
    "vocab_word",
)

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

EMBEDDING_EXTENSIONS = {".bin", ".vec", ".txt"}
GUIDED_UPLOAD_TARGETS = {"gold", "target"}
GUIDED_UPLOAD_DIR = Path("data/uploads/gui")
STATE_CHANGING_METHODS = {"POST", "PUT", "PATCH", "DELETE"}
PUBLIC_INFORMATION_ENDPOINTS = {"healthz", "impressum", "datenschutz", "static"}


class GUIError(ValueError):
    """Error type for request-level validation failures."""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = int(status_code)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return value


def _relative_to_root(path: Path, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def _resolve_repo_path(repo_root: Path, user_path: str | None, *, allow_nonexistent: bool) -> Path:
    """Resolve a user path and enforce containment beneath ``repo_root``.

    Resolution occurs before the containment check so absolute paths,
    traversal components, and symlinks cannot escape the trusted repository
    tree.  Callers may opt into paths that have not been created yet.
    """

    raw = "" if user_path is None else str(user_path).strip()
    if not raw:
        raise GUIError("Path is required.")

    candidate = Path(raw)
    resolved = candidate.resolve(strict=False) if candidate.is_absolute() else (repo_root / candidate).resolve(strict=False)
    try:
        resolved.relative_to(repo_root)
    except ValueError as exc:
        raise GUIError("Path must resolve inside the repository root.") from exc

    if not allow_nonexistent and not resolved.exists():
        raise GUIError(f"Path does not exist: {raw}")
    return resolved


def _resolve_json_path(repo_root: Path, user_path: str | None, *, allow_nonexistent: bool) -> Path:
    """Resolve a contained repository path and require a JSON filename."""

    path = _resolve_repo_path(repo_root, user_path, allow_nonexistent=allow_nonexistent)
    if path.suffix.lower() != ".json":
        raise GUIError("Config file path must end with .json")
    return path


def _load_raw_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise GUIError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise GUIError("Config JSON must be an object.")
    return payload


def _coerce_config_object(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise GUIError("Expected 'config' to be a JSON object.")
    return payload


def _sanitize_filename(name: str | None, fallback: str = "config.json") -> str:
    raw = (name or "").strip()
    if not raw:
        return fallback
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", raw)
    if not cleaned:
        return fallback
    if not cleaned.endswith(".json"):
        cleaned = f"{cleaned}.json"
    return cleaned


def _sanitize_upload_filename(name: str | None, fallback: str = "upload.txt") -> str:
    """Reduce an upload name to a bounded basename with a safe extension."""

    raw_name = Path(str(name or "").strip()).name
    if not raw_name:
        raw_name = fallback

    suffix = Path(raw_name).suffix
    stem = Path(raw_name).stem or Path(fallback).stem or "upload"
    stem_clean = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-") or "upload"
    suffix_clean = re.sub(r"[^A-Za-z0-9.]+", "", suffix)[:12]
    if suffix_clean and not suffix_clean.startswith("."):
        suffix_clean = f".{suffix_clean}"
    if not suffix_clean:
        fallback_suffix = Path(fallback).suffix
        suffix_clean = fallback_suffix if fallback_suffix else ".txt"
    return f"{stem_clean}{suffix_clean}"


def _unique_path_for_write(path: Path) -> Path:
    """Select a non-existing sibling name without overwriting user data."""

    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    for idx in range(2, 10_000):
        candidate = path.with_name(f"{stem}_{idx}{suffix}")
        if not candidate.exists():
            return candidate
    raise GUIError("Could not reserve a unique output filename.", status_code=500)


def _embedding_candidates(repo_root: Path) -> list[dict[str, str]]:
    """List supported in-repository embeddings with unambiguous labels.

    Paths remain repository-relative, and duplicate filenames are qualified by
    parent directory and a deterministic suffix where necessary.
    """

    embeddings_root = (repo_root / "embeddings").resolve(strict=False)
    try:
        embeddings_root.relative_to(repo_root)
    except ValueError:
        return []
    if not embeddings_root.exists() or not embeddings_root.is_dir():
        return []

    items: list[dict[str, str]] = []
    for candidate in embeddings_root.rglob("*"):
        if not candidate.is_file():
            continue
        if candidate.suffix.lower() not in EMBEDDING_EXTENSIONS:
            continue
        items.append(
            {
                "name": candidate.name,
                "parent": candidate.parent.name,
                "path": _relative_to_root(candidate, repo_root),
            }
        )

    label_counts: dict[str, int] = {}
    for item in items:
        key = item["name"]
        label_counts[key] = label_counts.get(key, 0) + 1

    label_with_parent_counts: dict[str, int] = {}
    labeled: list[dict[str, str]] = []
    for item in items:
        if label_counts[item["name"]] == 1:
            base_label = item["name"]
        else:
            base_label = f"{item['name']} ({item['parent']})"
        label_with_parent_counts[base_label] = label_with_parent_counts.get(base_label, 0) + 1
        labeled.append({"path": item["path"], "name": item["name"], "base_label": base_label})

    disambiguated_seen: dict[str, int] = {}
    result: list[dict[str, str]] = []
    for item in labeled:
        label = item["base_label"]
        if label_with_parent_counts[label] > 1:
            disambiguated_seen[label] = disambiguated_seen.get(label, 0) + 1
            label = f"{label} [{disambiguated_seen[label]}]"
        result.append({"label": label, "name": item["name"], "path": item["path"]})

    result.sort(key=lambda row: (row["label"].lower(), row["path"].lower()))
    return result


def _normalize_config_paths(config: dict[str, Any], repo_root: Path, task: str) -> dict[str, Any]:
    """Canonicalize task-relevant config paths within the repository boundary.

    The returned deep copy contains absolute paths suitable for core execution;
    the editor-owned input object remains unchanged. Existing inputs are
    required, while output locations may be created by the workflow.
    """
    cfg = copy.deepcopy(config)

    def set_resolved(keys: tuple[str, ...], allow_nonexistent: bool) -> None:
        node: Any = cfg
        for key in keys[:-1]:
            if not isinstance(node, dict):
                return
            node = node.get(key)
        if not isinstance(node, dict):
            return
        leaf = keys[-1]
        value = node.get(leaf)
        if value in (None, ""):
            return
        resolved = _resolve_repo_path(
            repo_root,
            str(value),
            allow_nonexistent=allow_nonexistent,
        )
        node[leaf] = str(resolved)

    set_resolved(("dataset", "gold"), allow_nonexistent=False)
    set_resolved(("runtime", "output_dir"), allow_nonexistent=True)

    if task == "prediction_run":
        spaces = cfg.get("embeddings", {}).get("spaces", [])
        if isinstance(spaces, list):
            for idx, space in enumerate(spaces):
                if not isinstance(space, dict):
                    continue
                path_value = space.get("path")
                if path_value in (None, ""):
                    continue
                space["path"] = str(
                    _resolve_repo_path(
                        repo_root,
                        str(path_value),
                        allow_nonexistent=False,
                    )
                )

        set_resolved(("prediction", "target"), allow_nonexistent=False)
        set_resolved(("prediction", "holdout"), allow_nonexistent=True)
        set_resolved(("prediction", "predictions_csv"), allow_nonexistent=True)
        set_resolved(("prediction", "unmatched_csv"), allow_nonexistent=True)

    if task == "prediction_holdout":
        set_resolved(("prediction", "holdout"), allow_nonexistent=False)
        set_resolved(("prediction", "predictions_csv"), allow_nonexistent=False)
        set_resolved(("prediction", "unmatched_csv"), allow_nonexistent=True)

    return cfg


def _parse_pagination(raw_offset: str | None, raw_limit: str | None) -> tuple[int, int]:
    try:
        offset = int(raw_offset or 0)
    except ValueError:
        raise GUIError("offset must be an integer.")
    try:
        limit = int(raw_limit or DEFAULT_RESULT_PREVIEW_LIMIT)
    except ValueError:
        raise GUIError("limit must be an integer.")

    if offset < 0:
        raise GUIError("offset must be >= 0.")
    if limit < 1:
        raise GUIError("limit must be >= 1.")
    return offset, min(limit, MAX_RESULT_PREVIEW_LIMIT)


def _pick_word_search_column(columns: list[str]) -> str | None:
    by_lower: dict[str, str] = {}
    for column in columns:
        normalized = str(column).strip().lower()
        if normalized and normalized not in by_lower:
            by_lower[normalized] = str(column)
    for candidate in WORD_SEARCH_COLUMN_CANDIDATES:
        if candidate in by_lower:
            return by_lower[candidate]
    return None


def _preview_csv(path: Path, query: str, offset: int, limit: int, *, apply_query: bool = True) -> dict[str, Any]:
    """Scan a CSV once while retaining only the requested preview window.

    Matching rows are counted across the complete file so pagination metadata
    remains exact, but memory use is bounded by ``limit``. When enabled, search
    is restricted to the first recognized word-like column.
    """

    query_lc = query.lower() if apply_query else ""
    rows: list[dict[str, str]] = []
    columns: list[str] = []
    total_matches = 0

    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        word_search_column = _pick_word_search_column(columns) if query_lc else None
        for row_index, row in enumerate(reader):
            ordered_row = {column: str(row.get(column, "")) for column in columns}
            if query_lc:
                if not word_search_column:
                    continue
                if query_lc not in str(ordered_row.get(word_search_column, "")).lower():
                    continue

            if offset <= total_matches < offset + limit:
                rows.append({"_row_index": str(row_index), **ordered_row})
            total_matches += 1

    return {
        "kind": "csv",
        "columns": ["_row_index", *columns],
        "rows": rows,
        "offset": offset,
        "limit": limit,
        "total_matches": total_matches,
        "has_more": (offset + limit) < total_matches,
    }


def _preview_text(path: Path, query: str, offset: int, limit: int) -> dict[str, Any]:
    """Return an exact, memory-bounded page from a streamed text artifact."""

    query_lc = query.lower()
    window: list[tuple[int, str]] = []
    total_matches = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_idx, raw_line in enumerate(handle, start=1):
            line = raw_line.rstrip("\r\n")
            if query_lc and query_lc not in line.lower():
                continue
            if offset <= total_matches < offset + limit:
                window.append((line_idx, line))
            total_matches += 1

    return {
        "kind": "text",
        "lines": [{"line_number": line_no, "text": text} for line_no, text in window],
        "offset": offset,
        "limit": limit,
        "total_matches": total_matches,
        "has_more": (offset + limit) < total_matches,
    }


def _config_from_payload(
    payload: dict[str, Any],
    repo_root: Path,
) -> tuple[dict[str, Any], str]:
    """Load a live editor config or a repository config referenced by path.

    Live editor data are merged with canonical defaults. Path-based configs
    pass through the core loader and the repository containment policy.
    """

    if isinstance(payload.get("config"), dict):
        raw = _coerce_config_object(payload["config"])
        merged = _deep_merge(core_default_config(), raw)
        config_path = str(payload.get("config_path", "<gui_inline_config>"))
        return merged, config_path

    config_path = _resolve_json_path(
        repo_root,
        payload.get("config_path"),
        allow_nonexistent=False,
    )
    merged = core_load_config(str(config_path))
    return merged, str(config_path)


def _discover_existing_artifacts(
    config: dict[str, Any],
    repo_root: Path,
) -> tuple[dict[str, Path], list[str]]:
    """Best-effort discovery of existing prediction/holdout artifacts on disk."""
    warnings: list[str] = []
    artifacts: dict[str, Path] = {}

    runtime_cfg = config.get("runtime")
    prediction_cfg = config.get("prediction")
    reports_cfg = config.get("reports")
    if not isinstance(runtime_cfg, dict) or not isinstance(prediction_cfg, dict):
        return artifacts, warnings

    out_dir_raw = runtime_cfg.get("output_dir")
    if out_dir_raw in (None, ""):
        return artifacts, warnings

    try:
        out_dir = _resolve_repo_path(repo_root, str(out_dir_raw), allow_nonexistent=True)
    except GUIError as exc:
        warnings.append(f"Could not inspect existing results: {exc}")
        return {}, warnings

    def maybe_add_in_out_dir(artifact_key: str) -> None:
        filename = ARTIFACT_FILENAMES.get(artifact_key)
        if filename is None:
            return
        path = out_dir / filename
        if path.exists() and path.is_file():
            artifacts[artifact_key] = path

    maybe_add_in_out_dir("summary")
    maybe_add_in_out_dir("cv_results")
    maybe_add_in_out_dir("test_predictions")
    maybe_add_in_out_dir("holdout_summary")
    maybe_add_in_out_dir("holdout_unmatched")

    report_level = str((reports_cfg or {}).get("level", "core"))
    if report_level == "full":
        maybe_add_in_out_dir("oov_gold")
        maybe_add_in_out_dir("oov_vocab")
        if prediction_cfg.get("target") not in (None, ""):
            maybe_add_in_out_dir("vocab_predictions")

    unmatched_raw = prediction_cfg.get("unmatched_csv")
    if unmatched_raw not in (None, ""):
        try:
            unmatched_path = _resolve_repo_path(
                repo_root,
                str(unmatched_raw),
                allow_nonexistent=True,
            )
        except GUIError as exc:
            warnings.append(f"Could not inspect existing results: {exc}")
            return {}, warnings
        if unmatched_path.exists() and unmatched_path.is_file():
            artifacts["holdout_unmatched"] = unmatched_path

    return artifacts, warnings


def _existing_results_signature(
    *,
    config: dict[str, Any],
    config_path: str,
    artifacts: dict[str, Path],
) -> str:
    """Serialize config and artifact identity for idempotent registration."""

    artifact_items = [[str(key), str(path)] for key, path in sorted(artifacts.items())]
    payload = {
        "config_path": str(config_path),
        "config": _to_jsonable(config),
        "artifacts": artifact_items,
    }
    return json.dumps(payload, sort_keys=True)


def create_app(
    *,
    repo_root: Path | str | None = None,
    testing: bool = False,
    job_store: SQLiteJobStore | None = None,
    job_executor: Callable[[Any, Any, Any, Any], int] | None = None,
    start_embedded_worker: bool | None = None,
) -> Flask:
    """Create the Flask application and its durable job interfaces.

    Hosted behavior is derived from environment-backed :class:`HostedSettings`.
    Local development starts an embedded worker over the same SQLite queue;
    production hosted mode expects the external worker entrypoint. Tests may
    inject a store, executor, and explicit embedded-worker policy without
    changing route behavior.
    """

    app = Flask(__name__, template_folder="templates", static_folder="static")
    effective_root = Path(repo_root).resolve() if repo_root is not None else DEFAULT_REPO_ROOT
    hosted = HostedSettings.from_env(repo_root=effective_root)
    app.config["GUI_REPO_ROOT"] = effective_root
    app.config["TESTING"] = bool(testing)
    app.config["HOSTED_SETTINGS"] = hosted
    app.config["MAX_CONTENT_LENGTH"] = hosted.limits.max_upload_bytes + (256 * 1024)
    app.config["SESSION_COOKIE_NAME"] = "concritup_session"
    app.config["SESSION_COOKIE_HTTPONLY"] = True
    app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
    app.config["SESSION_COOKIE_SECURE"] = bool(hosted.enabled)
    app.config["SECRET_KEY"] = hosted.secret_key or secrets.token_hex(32)

    jobs = job_store or SQLiteJobStore(
        hosted.db_path,
        hosted.job_root,
        global_queue_limit=hosted.limits.global_jobs,
        per_owner_queue_limit=hosted.limits.per_owner_jobs,
        storage_stop_fraction=hosted.limits.storage_stop_fraction,
        redact_roots=(effective_root, hosted.session_root),
    )
    app.extensions["concritup_job_store"] = jobs

    entrypoint = os.environ.get("CONCRITUP_CORE_ENTRYPOINT", "").strip()
    if not entrypoint:
        local_entrypoint = effective_root / ".venv" / "bin" / "concreteness-knn-core"
        entrypoint = str(local_entrypoint if local_entrypoint.exists() else "concreteness-knn-core")
    worker_settings = WorkerSettings(
        repo_root=effective_root,
        db_path=hosted.db_path,
        job_root=hosted.job_root,
        entrypoint=entrypoint,
        worker_id=f"embedded-{os.getpid()}-{id(app)}",
        session_root=hosted.session_root if hosted.enabled else None,
        retention_hours=hosted.limits.retention_hours,
        global_queue_limit=hosted.limits.global_jobs,
        per_owner_queue_limit=hosted.limits.per_owner_jobs,
        storage_stop_fraction=hosted.limits.storage_stop_fraction,
    )

    def default_embedded_executor(job: Any, _command: Any, _cwd: Any, _environment: Any) -> int:
        config = core_load_config(str(job.paths.config_path))
        with job.paths.log_path.open("a", encoding="utf-8") as log_handle:
            with contextlib.redirect_stdout(log_handle), contextlib.redirect_stderr(log_handle):
                if job.command == "prediction-run":
                    PredictionPipeline().run_prediction(
                        config,
                        config_path=str(job.paths.config_path),
                    )
                else:
                    PredictionPipeline().run_holdout(
                        config,
                        config_path=str(job.paths.config_path),
                    )
        return 0

    embedded_requested = (
        (not hosted.enabled) if start_embedded_worker is None else bool(start_embedded_worker)
    )
    if job_executor is not None:
        embedded_requested = True
    embedded_worker: JobWorker | None = None
    if embedded_requested:
        embedded_worker = JobWorker(
            jobs,
            worker_settings,
            executor=job_executor or (default_embedded_executor if testing else None),
        )
        embedded_worker.recover()
        if not testing:
            worker_thread = threading.Thread(
                target=embedded_worker.run_forever,
                kwargs={"recover": False},
                name="concritup-embedded-worker",
                daemon=True,
            )
            worker_thread.start()
            app.extensions["concritup_worker_thread"] = worker_thread
    app.extensions["concritup_embedded_worker"] = embedded_worker

    def kick_embedded_worker() -> None:
        if embedded_worker is None or not testing:
            return
        threading.Thread(
            target=embedded_worker.run_once,
            name="concritup-test-worker",
            daemon=True,
        ).start()

    def ok(payload: dict[str, Any], status: int = 200):
        return jsonify(payload), status

    def fail(exc: Exception):
        message = (
            sanitize_public_message(
                exc,
                redact_roots=(effective_root, hosted.job_root, hosted.session_root),
            )
            if hosted.enabled
            else str(exc)
        )
        if isinstance(exc, GUIError):
            return ok({"error": message}, status=exc.status_code)
        if isinstance(exc, (HostedConfigError, ValueError, json.JSONDecodeError)):
            return ok({"error": message}, status=400)
        if isinstance(exc, JobAdmissionError):
            return ok({"error": message}, status=409)
        return ok({"error": message}, status=500)

    def request_owner() -> str:
        owner = getattr(g, "concritup_owner", None)
        if not owner:
            raise GUIError("Request session is unavailable.", status_code=401)
        return str(owner)

    def path_resolver() -> HostedPathResolver:
        resolver = getattr(g, "concritup_paths", None)
        if resolver is None:
            raise GUIError("Hosted path resolver is unavailable.", status_code=500)
        return resolver

    @app.errorhandler(RequestEntityTooLarge)
    def upload_too_large(_exc: RequestEntityTooLarge):
        return ok(
            {
                "error": (
                    f"Uploaded files must not exceed "
                    f"{hosted.limits.max_upload_bytes // (1024 * 1024)} MiB."
                )
            },
            status=413,
        )

    @app.before_request
    def initialize_request_identity():
        if request.endpoint in PUBLIC_INFORMATION_ENDPOINTS:
            return None

        if not hosted.enabled or hosted.access_mode == "trusted":
            owner = "trusted"
        elif hosted.access_mode == "authenticated":
            remote_user = str(request.headers.get("X-Remote-User", "")).strip()
            if not remote_user or len(remote_user) > 200:
                return ok({"error": "Authenticated staging identity is required."}, status=401)
            owner = "authenticated:" + hashlib.sha256(remote_user.encode("utf-8")).hexdigest()
        else:
            owner = str(session.get("anonymous_owner", ""))
            if not re.fullmatch(r"[0-9a-f]{64}", owner):
                owner = secrets.token_hex(32)
                session["anonymous_owner"] = owner

        g.concritup_owner = owner
        if hosted.enabled:
            resolver = HostedPathResolver(hosted, owner)
            session_lease = resolver.acquire_owner_storage(create_owner=False)
            g.concritup_paths = resolver
            g.concritup_session_lease = session_lease

        if hosted.enabled and hosted.access_mode in {"authenticated", "anonymous"}:
            csrf_token = str(session.get("csrf_token", ""))
            if not csrf_token:
                csrf_token = secrets.token_urlsafe(32)
                session["csrf_token"] = csrf_token
            if request.method in STATE_CHANGING_METHODS:
                supplied = str(request.headers.get("X-CSRF-Token", ""))
                if not supplied or not hmac.compare_digest(supplied, csrf_token):
                    return ok({"error": "Invalid or missing CSRF token."}, status=403)
        return None

    @app.teardown_request
    def release_request_session_storage(_error: BaseException | None) -> None:
        """Release the cross-process owner lock held for one hosted request."""

        lease = getattr(g, "concritup_session_lease", None)
        if lease is not None:
            lease.close()

    def artifact_alias(job_id: str, artifact_key: str, path: Path) -> str:
        if hosted.enabled:
            return f"job-artifact:{job_id}:{artifact_key}"
        return _relative_to_root(path, effective_root)

    def public_job(entry: dict[str, Any]) -> dict[str, Any]:
        result = copy.deepcopy(entry.get("result"))
        if isinstance(result, dict) and isinstance(result.get("artifacts"), dict):
            result["artifacts"] = {
                str(key): artifact_alias(
                    str(entry["job_id"]),
                    str(key),
                    Path(str(value)).resolve(strict=False),
                )
                for key, value in result["artifacts"].items()
            }
            if hosted.enabled:
                result["config_path"] = f"job-config:{entry['job_id']}"
        return {
            "job_id": str(entry.get("job_id", "")),
            "command": str(entry.get("command", "")),
            "payload": copy.deepcopy(entry.get("payload") or {}),
            "status": "queued" if entry.get("status") == "preparing" else entry.get("status"),
            "created_at": entry.get("created_at"),
            "updated_at": entry.get("updated_at"),
            "logs": list(entry.get("logs") or []),
            "result": result,
            "error": entry.get("error"),
        }

    def reject_if_busy():
        active = jobs.active_job(owner=request_owner())
        if active is None:
            return None
        active_public = public_job(active)
        return ok(
            {
                "error": "Another job is already queued or running.",
                "active_job": active_public,
            },
            status=409,
        )

    def artifact_path_for_job(job_id: str, artifact_key: str) -> tuple[dict[str, Any], Path]:
        entry = jobs.get(job_id, owner=request_owner())
        if entry is None:
            raise GUIError(f"Unknown job id: {job_id}", status_code=404)
        if entry.get("status") != "done":
            raise GUIError("Results are only available for completed jobs.")
        result = entry.get("result")
        if not isinstance(result, dict):
            raise GUIError("Completed job has no result payload.")
        artifacts = result.get("artifacts")
        if not isinstance(artifacts, dict):
            raise GUIError("Completed job has no artifact map.")
        if artifact_key not in artifacts:
            raise GUIError(f"Unknown artifact_key '{artifact_key}' for job '{job_id}'.", status_code=404)
        if hosted.enabled:
            artifact_path = Path(str(artifacts[artifact_key])).resolve(strict=False)
            job_root = jobs.paths_for(job_id).job_dir.resolve(strict=False)
            try:
                artifact_path.relative_to(job_root)
            except ValueError as exc:
                raise GUIError("Artifact path is outside its job bundle.", status_code=404) from exc
            if not artifact_path.is_file():
                raise GUIError("Artifact no longer exists.", status_code=404)
        else:
            artifact_path = _resolve_repo_path(
                app.config["GUI_REPO_ROOT"],
                str(artifacts[artifact_key]),
                allow_nonexistent=False,
            )
        return entry, artifact_path

    def discover_and_register_existing_results(
        config: dict[str, Any],
        *,
        config_path: str,
        source: str,
    ) -> dict[str, Any]:
        if hosted.enabled:
            return {
                "found": False,
                "job_id": None,
                "artifacts": {},
                "warnings": [],
            }
        warnings: list[str] = []
        try:
            artifacts, warnings = _discover_existing_artifacts(config, app.config["GUI_REPO_ROOT"])
        except Exception as exc:  # pragma: no cover - defensive
            warnings = [f"Could not inspect existing results: {exc}"]
            artifacts = {}

        if not artifacts:
            return {
                "found": False,
                "job_id": None,
                "artifacts": {},
                "warnings": warnings,
            }

        owner = request_owner()
        signature_source = _existing_results_signature(
            config=config,
            config_path=config_path,
            artifacts=artifacts,
        )
        signature = hashlib.sha256(
            f"{owner}\0{signature_source}".encode("utf-8")
        ).hexdigest()
        result_payload = {
            "config_path": str(config_path),
            "artifacts": {str(key): str(path) for key, path in artifacts.items()},
        }
        try:
            job_id = jobs.register_existing_result(
                command="existing-results",
                payload={"source": str(source), "config_path": str(config_path)},
                result=result_payload,
                signature=signature,
                owner=owner,
            )
        except Exception as exc:  # pragma: no cover - defensive
            warnings.append(f"Could not attach existing results: {exc}")
            return {
                "found": False,
                "job_id": None,
                "artifacts": {},
                "warnings": warnings,
            }

        relative_artifacts = {
            str(key): _relative_to_root(path, app.config["GUI_REPO_ROOT"])
            for key, path in artifacts.items()
        }
        return {
            "found": True,
            "job_id": str(job_id),
            "artifacts": relative_artifacts,
            "warnings": warnings,
        }

    def no_existing_results() -> dict[str, Any]:
        return {"found": False, "job_id": None, "artifacts": {}, "warnings": []}

    def publicize_hosted_config(config: dict[str, Any]) -> dict[str, Any]:
        if not hosted.enabled:
            return config
        cfg = copy.deepcopy(config)
        public_entries = {
            str(item["id"]): item for item in hosted.embedding_allowlist.public_entries()
        }
        embeddings = cfg.get("embeddings")
        if not isinstance(embeddings, dict) or not public_entries:
            return cfg
        active_id = str(embeddings.get("active_space", ""))
        selected = public_entries.get(active_id)
        if selected is None:
            selected = public_entries[sorted(public_entries)[0]]
        embeddings["mode"] = "single"
        embeddings["active_space"] = selected["id"]
        embeddings["spaces"] = [
            {
                "id": selected["id"],
                "kind": selected["kind"],
                "path": selected["path"],
                "label": selected["label"],
            }
        ]
        cfg.setdefault("runtime", {})["n_jobs"] = 1
        return cfg

    def resolve_hosted_config_path(raw_path: object, *, for_write: bool) -> tuple[Path, str]:
        raw = str(raw_path or "").strip()
        if not raw:
            raise GUIError("Path is required.")
        resolver = path_resolver()
        if raw.startswith("paper-config:"):
            if for_write:
                raise GUIError("Paper configurations are read-only.", status_code=403)
            path = resolver.resolve(raw, must_exist=True)
            return path, raw
        if raw.startswith("session-config:"):
            path = resolver.resolve(raw, must_exist=not for_write)
            return path, raw

        candidate = Path(raw)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise GUIError("Hosted config paths must use the virtual config browser.")
        parts = candidate.parts
        if len(parts) == 3 and parts[:2] == ("configs", "paper_runs"):
            if for_write:
                raise GUIError("Paper configurations are read-only.", status_code=403)
            alias = f"paper-config:{parts[2]}"
        elif len(parts) == 3 and parts[:2] == ("configs", "user"):
            alias = f"session-config:{parts[2]}"
        elif len(parts) == 2 and parts[0] == "configs":
            alias = f"session-config:{parts[1]}"
        elif len(parts) == 1:
            alias = f"session-config:{parts[0]}"
        else:
            raise GUIError("Unknown virtual config path.")
        path = resolver.resolve(alias, must_exist=not for_write)
        return path, alias

    def read_upload_bytes(uploaded: Any) -> bytes:
        limit = hosted.limits.max_upload_bytes if hosted.enabled else 10 * 1024 * 1024
        payload = uploaded.stream.read(limit + 1)
        if len(payload) > limit:
            raise GUIError(f"Uploaded files must not exceed {limit // (1024 * 1024)} MiB.", status_code=413)
        return payload

    def require_storage_admission() -> None:
        if hosted.enabled and not jobs.admission_available():
            percent = round(hosted.limits.storage_stop_fraction * 100)
            raise JobAdmissionError(
                f"Hosted storage is at or above the {percent}% admission threshold."
            )

    def resolve_hosted_input_alias(value: object) -> Path:
        alias = str(value or "").strip()
        if alias.startswith("upload:"):
            return path_resolver().resolve(alias, must_exist=True)
        if alias.startswith("job-artifact:"):
            parts = alias.split(":", 2)
            if len(parts) != 3 or not parts[1] or not parts[2]:
                raise GUIError("Invalid job artifact alias.")
            _entry, path = artifact_path_for_job(parts[1], parts[2])
            return path
        raise GUIError(
            "Hosted data paths must refer to a session upload or one of this session's job artifacts."
        )

    @app.route("/healthz")
    def healthz():
        return Response("ok\n", status=200, mimetype="text/plain")

    @app.route("/")
    def index() -> Response | str:
        return render_template(
            "index.html",
            repo_root=str(app.config["GUI_REPO_ROOT"]),
            csrf_token=str(session.get("csrf_token", "")),
            hosted_enabled=hosted.enabled,
            retention_hours=f"{hosted.limits.retention_hours:g}",
        )

    @app.route("/impressum")
    def impressum() -> str:
        return render_template("impressum.html")

    @app.route("/datenschutz")
    def datenschutz() -> str:
        return render_template(
            "datenschutz.html",
            retention_hours=f"{hosted.limits.retention_hours:g}",
        )

    @app.route("/api/config/default", methods=["GET"])
    def api_config_default():
        try:
            raw = publicize_hosted_config(core_default_config())
            return ok(
                {
                    "source": "default_config",
                    "raw_config": raw,
                    "resolved_config": raw,
                }
            )
        except Exception as exc:  # pragma: no cover
            return fail(exc)

    @app.route("/api/config/load", methods=["POST"])
    def api_config_load():
        try:
            payload = request.get_json(silent=True) or {}
            if hosted.enabled:
                path, rel_path = resolve_hosted_config_path(payload.get("path"), for_write=False)
            else:
                path = _resolve_json_path(
                    app.config["GUI_REPO_ROOT"],
                    payload.get("path"),
                    allow_nonexistent=False,
                )
                rel_path = _relative_to_root(path, app.config["GUI_REPO_ROOT"])
            raw = _load_raw_json(path)
            resolved = core_load_config(str(path))
            raw = publicize_hosted_config(raw)
            resolved = publicize_hosted_config(resolved)
            existing_results = discover_and_register_existing_results(
                resolved,
                config_path=rel_path,
                source="config_load",
            )
            return ok(
                {
                    "path": rel_path,
                    "raw_config": raw,
                    "resolved_config": resolved,
                    "existing_results": existing_results,
                }
            )
        except Exception as exc:
            return fail(exc)

    @app.route("/api/config/save", methods=["POST"])
    def api_config_save():
        try:
            payload = request.get_json(silent=True) or {}
            config = _coerce_config_object(payload.get("config"))
            if hosted.enabled:
                require_storage_admission()
                path_resolver().activate_owner_storage()
                path, response_path = resolve_hosted_config_path(payload.get("path"), for_write=True)
            else:
                path = _resolve_json_path(
                    app.config["GUI_REPO_ROOT"],
                    payload.get("path"),
                    allow_nonexistent=True,
                )
                response_path = _relative_to_root(path, app.config["GUI_REPO_ROOT"])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
            resolved = core_load_config(str(path))
            return ok(
                {
                    "path": response_path,
                    "raw_config": config,
                    "resolved_config": publicize_hosted_config(resolved),
                }
            )
        except Exception as exc:
            return fail(exc)

    @app.route("/api/config/upload", methods=["POST"])
    def api_config_upload():
        try:
            if "file" not in request.files:
                raise GUIError("Multipart field 'file' is required.")
            uploaded = request.files["file"]
            if uploaded.filename is None:
                raise GUIError("Uploaded file name is missing.")
            text = read_upload_bytes(uploaded).decode("utf-8")
            payload = json.loads(text)
            raw = _coerce_config_object(payload)
            resolved = _deep_merge(core_default_config(), raw)
            raw = publicize_hosted_config(raw)
            upload_label = _sanitize_filename(uploaded.filename, fallback="config.json")
            resolved = publicize_hosted_config(resolved)
            existing_results = (
                no_existing_results()
                if hosted.enabled
                else discover_and_register_existing_results(
                    resolved,
                    config_path=f"<uploaded:{upload_label}>",
                    source="config_upload",
                )
            )
            return ok(
                {
                    "filename": uploaded.filename,
                    "raw_config": raw,
                    "resolved_config": resolved,
                    "existing_results": existing_results,
                }
            )
        except Exception as exc:
            return fail(exc)

    @app.route("/api/fs/upload", methods=["POST"])
    def api_fs_upload():
        try:
            if "file" not in request.files:
                raise GUIError("Multipart field 'file' is required.")
            uploaded = request.files["file"]
            if uploaded.filename is None:
                raise GUIError("Uploaded file name is missing.")

            field_target = str(request.form.get("field_target", "")).strip()
            if field_target not in GUIDED_UPLOAD_TARGETS:
                raise GUIError("field_target must be one of: gold, target")

            fallback = "gold.csv" if field_target == "gold" else "target.txt"
            safe_name = _sanitize_upload_filename(uploaded.filename, fallback=fallback)
            payload = read_upload_bytes(uploaded)
            require_storage_admission()
            if hosted.enabled:
                resolver = path_resolver()
                resolver.activate_owner_storage()
                upload_dir = resolver.upload_root
            else:
                upload_dir = _resolve_repo_path(
                    app.config["GUI_REPO_ROOT"],
                    str(GUIDED_UPLOAD_DIR),
                    allow_nonexistent=True,
                )
            upload_dir.mkdir(parents=True, exist_ok=True)
            output_path = _unique_path_for_write(upload_dir / safe_name)
            output_path.write_bytes(payload)
            response_path = (
                path_resolver().alias_for(output_path)
                if hosted.enabled
                else _relative_to_root(output_path, app.config["GUI_REPO_ROOT"])
            )

            return ok(
                {
                    "field_target": field_target,
                    "filename": uploaded.filename,
                    "path": response_path,
                },
                status=201,
            )
        except Exception as exc:
            return fail(exc)

    @app.route("/api/config/download", methods=["GET"])
    def api_config_download():
        try:
            maybe_path = request.args.get("path", "").strip()
            filename = _sanitize_filename(request.args.get("filename"), fallback="config.json")

            if maybe_path:
                if hosted.enabled:
                    path, _alias = resolve_hosted_config_path(maybe_path, for_write=False)
                else:
                    path = _resolve_json_path(
                        app.config["GUI_REPO_ROOT"],
                        maybe_path,
                        allow_nonexistent=False,
                    )
                return send_file(
                    str(path),
                    as_attachment=True,
                    download_name=filename,
                    mimetype="application/json",
                )

            raw_config = request.args.get("config", "").strip()
            if not raw_config:
                raise GUIError("Provide either query parameter 'path' or 'config'.")
            payload = json.loads(raw_config)
            raw = _coerce_config_object(payload)
            body = json.dumps(raw, indent=2) + "\n"
            return Response(
                body,
                mimetype="application/json",
                headers={"Content-Disposition": f'attachment; filename="{filename}"'},
            )
        except Exception as exc:
            return fail(exc)

    @app.route("/api/fs/embeddings", methods=["GET"])
    def api_fs_embeddings():
        try:
            if hosted.enabled:
                candidates = [
                    {
                        "label": str(item["label"]),
                        "name": str(item["filename"]),
                        "path": str(item["path"]),
                    }
                    for item in hosted.embedding_allowlist.public_entries()
                ]
            else:
                candidates = _embedding_candidates(app.config["GUI_REPO_ROOT"])
            return ok({"candidates": candidates})
        except Exception as exc:
            return fail(exc)

    @app.route("/api/fs/list", methods=["GET"])
    def api_fs_list():
        try:
            relative = request.args.get("path", ".")
            if hosted.enabled:
                return ok(path_resolver().virtual_listing(relative))
            base = _resolve_repo_path(
                app.config["GUI_REPO_ROOT"],
                relative,
                allow_nonexistent=False,
            )
            if not base.is_dir():
                raise GUIError("Path is not a directory.")

            entries: list[dict[str, Any]] = []
            for child in sorted(base.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower())):
                entries.append(
                    {
                        "name": child.name,
                        "path": _relative_to_root(child, app.config["GUI_REPO_ROOT"]),
                        "is_dir": child.is_dir(),
                        "is_file": child.is_file(),
                    }
                )

            parent_path: str | None = None
            if base != app.config["GUI_REPO_ROOT"]:
                parent_path = _relative_to_root(base.parent, app.config["GUI_REPO_ROOT"])

            return ok(
                {
                    "path": _relative_to_root(base, app.config["GUI_REPO_ROOT"]),
                    "parent_path": parent_path,
                    "entries": entries,
                }
            )
        except Exception as exc:
            return fail(exc)

    def copy_name(label: str, source: Path) -> str:
        suffix = source.suffix.lower()
        if not re.fullmatch(r"\.[a-z0-9]{1,12}", suffix):
            suffix = ".txt"
        return f"{label}{suffix}"

    def force_job_outputs(config: dict[str, Any], paths: JobPaths, *, task: str) -> None:
        config.setdefault("runtime", {})["output_dir"] = str(paths.output_dir)
        prediction = config.setdefault("prediction", {})
        prediction["unmatched_csv"] = str(paths.output_dir / "holdout_unmatched.csv")
        if task == "prediction-run" and prediction.get("target") not in (None, ""):
            prediction["predictions_csv"] = str(paths.output_dir / "vocab_predictions.csv")

    def hosted_submission_config(
        raw_config: dict[str, Any],
        *,
        task: str,
        paths: JobPaths,
    ) -> tuple[dict[str, Any], dict[str, Path]]:
        """Authorize, validate, and relocate inputs for one hosted job bundle.

        Virtual aliases are resolved within owner-managed storage before the
        hosted policy applies model limits and allowlisted resources. The
        resulting config points to deterministic job input copies, returned as
        a source mapping for atomic bundle preparation.
        """

        config = copy.deepcopy(raw_config)
        dataset = config.setdefault("dataset", {})
        prediction = config.setdefault("prediction", {})
        source_fields: list[tuple[dict[str, Any], str, str]] = [
            (dataset, "gold", "gold"),
        ]
        if task == "prediction-run":
            source_fields.append((prediction, "target", "target"))
            # Clear fields unused by prediction-run so the hosted manifest
            # cannot retain paths that were not authorized for this job.
            prediction["holdout"] = None
        else:
            source_fields.extend(
                [
                    (prediction, "holdout", "holdout"),
                    (prediction, "predictions_csv", "predictions"),
                ]
            )
            prediction["target"] = None

        allowed_roots: list[Path] = []
        for section, key, label in source_fields:
            value = section.get(key)
            if value in (None, ""):
                continue
            source = resolve_hosted_input_alias(value)
            section[key] = str(source)
            allowed_roots.append(source.parent)

        policy = HostedPolicy(hosted)
        config, _audit = policy.canonicalize_submission(
            config,
            job_paths=paths,
            allowed_data_roots=allowed_roots,
            task=task,
            validate_files=True,
        )

        input_files: dict[str, Path] = {}
        dataset = config["dataset"]
        prediction = config["prediction"]
        canonical_sections = {"gold": dataset, "target": prediction, "holdout": prediction, "predictions_csv": prediction}
        for _original_section, key, label in source_fields:
            section = canonical_sections[key]
            value = section.get(key)
            if value in (None, ""):
                continue
            source = Path(str(value)).resolve(strict=True)
            destination_name = copy_name(label, source)
            input_files[destination_name] = source
            section[key] = str(paths.input_dir / destination_name)

        force_job_outputs(config, paths, task=task)
        core_validate_config(
            config,
            task="prediction_run" if task == "prediction-run" else "prediction_holdout",
        )
        return config, input_files

    def submit_prediction_job(payload: dict[str, Any], *, task: str) -> str:
        config, _config_path = _config_from_payload(payload, app.config["GUI_REPO_ROOT"])
        job_id = jobs.new_job_id()
        paths = jobs.paths_for(job_id)
        input_files: dict[str, Path] = {}
        if hosted.enabled:
            config, input_files = hosted_submission_config(config, task=task, paths=paths)
        else:
            config = _normalize_config_paths(
                config,
                app.config["GUI_REPO_ROOT"],
                task="prediction_run" if task == "prediction-run" else "prediction_holdout",
            )
            force_job_outputs(config, paths, task=task)

        submitted = jobs.submit(
            task,
            copy.deepcopy(payload),
            owner=request_owner(),
            config=config,
            job_id=job_id,
            input_files=input_files,
            timeout_seconds=hosted.limits.timeout_seconds,
            output_limit_bytes=hosted.limits.output_limit_bytes,
        )
        kick_embedded_worker()
        return submitted

    @app.route("/api/jobs/prediction-run", methods=["POST"])
    def api_job_prediction_run():
        try:
            busy = reject_if_busy()
            if busy is not None:
                return busy
            payload = request.get_json(silent=True) or {}
            if not isinstance(payload.get("config"), dict):
                return ok({"error": "prediction-run requires live editor config in payload.config."}, status=400)
            job_id = submit_prediction_job(payload, task="prediction-run")
            return ok({"job_id": job_id, "status": "queued"}, status=202)
        except Exception as exc:
            return fail(exc)

    @app.route("/api/jobs/prediction-holdout", methods=["POST"])
    def api_job_prediction_holdout():
        try:
            busy = reject_if_busy()
            if busy is not None:
                return busy
            payload = request.get_json(silent=True) or {}
            if not isinstance(payload.get("config"), dict):
                return ok({"error": "prediction-holdout requires live editor config in payload.config."}, status=400)
            job_id = submit_prediction_job(payload, task="prediction-holdout")
            return ok({"job_id": job_id, "status": "queued"}, status=202)
        except Exception as exc:
            return fail(exc)

    @app.route("/api/jobs/<job_id>", methods=["GET"])
    def api_job_get(job_id: str):
        try:
            entry = jobs.get(job_id, owner=request_owner())
            if entry is None:
                raise GUIError(f"Unknown job id: {job_id}", status_code=404)
            return ok(public_job(entry))
        except Exception as exc:
            return fail(exc)

    @app.route("/api/results/latest", methods=["GET"])
    def api_results_latest():
        try:
            latest = jobs.latest_completed_job(owner=request_owner())
            if latest is None:
                return ok({"latest_job": None})

            result = latest.get("result")
            artifacts = {}
            if isinstance(result, dict) and isinstance(result.get("artifacts"), dict):
                for key, value in result["artifacts"].items():
                    path = Path(str(value)).resolve(strict=False)
                    if not path.is_file():
                        continue
                    artifacts[str(key)] = artifact_alias(
                        str(latest["job_id"]), str(key), path
                    )

            return ok(
                {
                    "latest_job": {
                        "job_id": latest.get("job_id"),
                        "command": latest.get("command"),
                        "status": latest.get("status"),
                        "created_at": latest.get("created_at"),
                        "updated_at": latest.get("updated_at"),
                        "artifacts": artifacts,
                    }
                }
            )
        except Exception as exc:
            return fail(exc)

    @app.route("/api/results/preview", methods=["GET"])
    def api_results_preview():
        try:
            job_id = str(request.args.get("job_id", "")).strip()
            artifact_key = str(request.args.get("artifact_key", "")).strip()
            query = str(request.args.get("q", "")).strip()
            if not job_id:
                raise GUIError("job_id is required.")
            if not artifact_key:
                raise GUIError("artifact_key is required.")
            offset, limit = _parse_pagination(
                request.args.get("offset"),
                request.args.get("limit"),
            )
            entry, artifact_path = artifact_path_for_job(job_id, artifact_key)
            suffix = artifact_path.suffix.lower()
            effective_query = query

            if suffix == ".csv":
                apply_query = artifact_key != "cv_results"
                if not apply_query:
                    effective_query = ""
                preview = _preview_csv(
                    artifact_path,
                    effective_query,
                    offset,
                    limit,
                    apply_query=apply_query,
                )
            else:
                preview = _preview_text(artifact_path, query, offset, limit)

            preview.update(
                {
                    "job_id": entry["job_id"],
                    "artifact_key": artifact_key,
                    "artifact_path": artifact_alias(job_id, artifact_key, artifact_path),
                    "query": effective_query,
                }
            )
            return ok(preview)
        except Exception as exc:
            return fail(exc)

    @app.route("/api/results/download", methods=["GET"])
    def api_results_download():
        try:
            job_id = str(request.args.get("job_id", "")).strip()
            artifact_key = str(request.args.get("artifact_key", "")).strip()
            if not job_id:
                raise GUIError("job_id is required.")
            if not artifact_key:
                raise GUIError("artifact_key is required.")
            _entry, artifact_path = artifact_path_for_job(job_id, artifact_key)
            return send_file(
                str(artifact_path),
                as_attachment=True,
                download_name=artifact_path.name,
            )
        except Exception as exc:
            return fail(exc)

    return app


if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=8000, debug=False)
