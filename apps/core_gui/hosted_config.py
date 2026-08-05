"""Hosted-mode limits, allowlisting, validation, and virtual path aliases.

The Flask-independent policy layer is shared by all hosted access modes.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from .job_store import JobPaths
from .session_storage import (
    SessionStorageError,
    SessionStorageLease,
    acquire_session_storage_lease,
    activate_session_storage,
    validate_session_root,
)


DEFAULT_POS_TOKEN_PATTERN = r"[,;/| ]+"
ACCESS_MODES = {"trusted", "authenticated", "anonymous"}


class HostedConfigError(ValueError):
    """A hosted request violates a server policy."""


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None or not value.strip():
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise HostedConfigError(f"Invalid boolean environment value: {value}")


def _as_nonnegative_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    try:
        value = default if not raw else float(raw)
    except ValueError as exc:
        raise HostedConfigError(f"{name} must be a finite non-negative number.") from exc
    if not math.isfinite(value) or value < 0:
        raise HostedConfigError(f"{name} must be a finite non-negative number.")
    return value


def _absolute_env_path(name: str, default: Path, repo_root: Path) -> Path:
    """Resolve relativity without following the configured final symlink."""

    raw = os.environ.get(name, "").strip()
    path = Path(raw) if raw else default
    if not path.is_absolute():
        path = repo_root / path
    return path.absolute()


def _resolved_env_path(name: str, default: Path, repo_root: Path) -> Path:
    return _absolute_env_path(name, default, repo_root).resolve(strict=False)


@dataclass(frozen=True)
class HostedLimits:
    """Resource and model-search limits enforced before job admission."""

    max_gold_rows: int = 40_000
    max_target_rows: int = 100_000
    max_upload_bytes: int = 10 * 1024 * 1024
    max_token_length: int = 100
    max_embedding_spaces: int = 1
    max_k: int = 100
    max_k_candidates: int = 20
    max_cv_folds: int = 5
    max_evidence_neighbors: int = 20
    per_owner_jobs: int = 1
    global_jobs: int = 20
    timeout_seconds: int = 120 * 60
    output_limit_bytes: int = 250 * 1024 * 1024
    retention_hours: float = 48.0
    storage_stop_fraction: float = 0.80


@dataclass(frozen=True)
class EmbeddingSpec:
    """Administrator-approved embedding metadata and resolved storage path."""

    id: str
    kind: str
    path: Path
    label: str

    def core_config(self) -> dict[str, str]:
        """Return the trusted embedding entry consumed by the core config."""

        return {
            "id": self.id,
            "kind": self.kind,
            "path": str(self.path),
            "label": self.label,
        }

    def public_config(self) -> dict[str, str]:
        """Return browser-safe metadata with no server filesystem path."""

        return {
            "id": self.id,
            "kind": self.kind,
            "path": f"embedding:{self.id}",
            "label": self.label,
            "filename": self.path.name,
        }


class EmbeddingAllowlist:
    """Administrator-owned mapping from public ids to filesystem paths."""

    def __init__(self, entries: Mapping[str, EmbeddingSpec]) -> None:
        self._entries = dict(entries)
        if len(self._entries) != len(set(self._entries)):
            raise HostedConfigError("Embedding allowlist ids must be unique.")

    @classmethod
    def empty(cls) -> "EmbeddingAllowlist":
        """Create an allowlist with no installed embeddings."""

        return cls({})

    @classmethod
    def from_payload(cls, payload: object, *, base_dir: Path) -> "EmbeddingAllowlist":
        """Validate and resolve allowlist entries from decoded JSON data.

        Relative administrator paths are resolved against ``base_dir`` and
        every listed embedding must already exist as a regular file.
        """

        if isinstance(payload, Mapping) and "embeddings" in payload:
            payload = payload["embeddings"]
        if isinstance(payload, Mapping):
            rows = []
            for key, value in payload.items():
                if isinstance(value, str):
                    rows.append({"id": key, "path": value})
                elif isinstance(value, Mapping):
                    rows.append({"id": key, **dict(value)})
                else:
                    raise HostedConfigError("Invalid embedding allowlist entry.")
        elif isinstance(payload, list):
            rows = payload
        else:
            raise HostedConfigError("Embedding allowlist must be a list or object.")

        entries: dict[str, EmbeddingSpec] = {}
        for idx, row in enumerate(rows):
            if not isinstance(row, Mapping):
                raise HostedConfigError(f"Embedding allowlist row {idx} must be an object.")
            identifier = str(row.get("id", "")).strip()
            if not re.fullmatch(r"[A-Za-z0-9._-]{1,80}", identifier):
                raise HostedConfigError(f"Invalid embedding id at row {idx}.")
            if identifier in entries:
                raise HostedConfigError(f"Duplicate embedding id: {identifier}")
            kind = str(row.get("kind", "ft_bin")).strip()
            if kind not in {"ft_bin", "vec"}:
                raise HostedConfigError(f"Embedding '{identifier}' has an unsupported kind.")
            raw_path = str(row.get("path", "")).strip()
            if not raw_path:
                raise HostedConfigError(f"Embedding '{identifier}' has no path.")
            path = Path(raw_path)
            if not path.is_absolute():
                path = base_dir / path
            path = path.resolve(strict=False)
            if not path.is_file():
                raise HostedConfigError(
                    f"Embedding '{identifier}' is not installed at its allowlisted path."
                )
            entries[identifier] = EmbeddingSpec(
                id=identifier,
                kind=kind,
                path=path,
                label=str(row.get("label", identifier)).strip() or identifier,
            )
        return cls(entries)

    @classmethod
    def from_source(cls, source: str, *, base_dir: Path) -> "EmbeddingAllowlist":
        """Load an allowlist from inline JSON or a JSON file path."""

        stripped = str(source).strip()
        if not stripped:
            return cls.empty()
        if stripped.startswith(("{", "[")):
            payload = json.loads(stripped)
            return cls.from_payload(payload, base_dir=base_dir)
        path = Path(stripped)
        if not path.is_absolute():
            path = base_dir / path
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_payload(payload, base_dir=path.parent.resolve())

    def get(self, identifier: str) -> EmbeddingSpec:
        """Return an approved embedding or raise a public policy error."""

        try:
            return self._entries[str(identifier)]
        except KeyError as exc:
            raise HostedConfigError("The selected embedding is not available on this server.") from exc

    def public_entries(self) -> list[dict[str, str]]:
        """Return deterministic browser-safe entries sorted by identifier."""

        return [self._entries[key].public_config() for key in sorted(self._entries)]

    def __len__(self) -> int:
        return len(self._entries)


@dataclass(frozen=True)
class HostedSettings:
    """Resolved web-hosting configuration and admission limits."""

    repo_root: Path
    enabled: bool
    access_mode: str
    secret_key: str
    db_path: Path
    job_root: Path
    session_root: Path
    paper_config_root: Path
    embedding_allowlist: EmbeddingAllowlist
    limits: HostedLimits = field(default_factory=HostedLimits)

    @classmethod
    def from_env(cls, *, repo_root: Path | str | None = None) -> "HostedSettings":
        """Build settings from ``CONCRITUP_*`` environment variables.

        Non-trusted hosted modes fail closed when the session secret is too
        short, and hosted execution requires at least one installed embedding.
        """

        effective_root = (
            Path(repo_root).resolve()
            if repo_root is not None
            else Path(__file__).resolve().parents[2]
        )
        enabled = _as_bool(os.environ.get("CONCRITUP_HOSTED_MODE"), False)
        access_mode = os.environ.get("CONCRITUP_ACCESS_MODE", "trusted").strip().lower()
        if access_mode not in ACCESS_MODES:
            raise HostedConfigError(
                "CONCRITUP_ACCESS_MODE must be trusted, authenticated, or anonymous."
            )
        secret_key = os.environ.get("CONCRITUP_SECRET_KEY", "")
        if enabled and access_mode != "trusted" and len(secret_key) < 32:
            raise HostedConfigError(
                "CONCRITUP_SECRET_KEY must contain at least 32 characters in hosted access modes."
            )
        allowlist_source = os.environ.get("CONCRITUP_EMBEDDING_ALLOWLIST", "")
        allowlist = EmbeddingAllowlist.from_source(allowlist_source, base_dir=effective_root)
        if enabled and len(allowlist) == 0:
            raise HostedConfigError("Hosted mode requires a non-empty embedding allowlist.")
        db_path = _resolved_env_path(
            "CONCRITUP_JOB_DB", effective_root / "data" / "job_queue.sqlite3", effective_root
        )
        job_root = _resolved_env_path(
            "CONCRITUP_JOB_ROOT", effective_root / "data" / "jobs", effective_root
        )
        session_candidate = _absolute_env_path(
            "CONCRITUP_SESSION_ROOT",
            effective_root / "data" / "hosted_sessions",
            effective_root,
        )
        try:
            session_root = validate_session_root(
                session_candidate,
                managed_data_root=effective_root / "data",
                disallowed_paths=(job_root, db_path),
            )
        except SessionStorageError as exc:
            raise HostedConfigError(str(exc)) from exc
        retention_hours = _as_nonnegative_float("CONCRITUP_RETENTION_HOURS", 48.0)
        return cls(
            repo_root=effective_root,
            enabled=enabled,
            access_mode=access_mode,
            secret_key=secret_key,
            db_path=db_path,
            job_root=job_root,
            session_root=session_root,
            paper_config_root=_resolved_env_path(
                "CONCRITUP_PAPER_CONFIG_ROOT", effective_root / "configs" / "paper_runs", effective_root
            ),
            embedding_allowlist=allowlist,
            limits=HostedLimits(retention_hours=retention_hours),
        )


def owner_storage_key(owner: str) -> str:
    """Derive a stable opaque directory key from a session owner identity."""

    normalized = str(owner).strip()
    if not normalized:
        raise HostedConfigError("Session owner is missing.")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


class HostedPathResolver:
    """Map browser-safe aliases to paper or owner-scoped files."""

    ROOT_ALIASES = ("configs", "uploads")

    def __init__(self, settings: HostedSettings, owner: str) -> None:
        self.settings = settings
        self.owner = str(owner)
        self.owner_root = settings.session_root / owner_storage_key(owner)
        self.config_root = self.owner_root / "configs"
        self.upload_root = self.owner_root / "uploads"

    @staticmethod
    def _safe_name(value: str, *, suffix: str | None = None) -> str:
        """Validate a single-component virtual filename and optional suffix."""

        name = Path(str(value)).name
        if name != str(value) or not re.fullmatch(r"[A-Za-z0-9._-]{1,255}", name):
            raise HostedConfigError("Invalid virtual filename.")
        if suffix is not None and Path(name).suffix.lower() != suffix:
            raise HostedConfigError(f"Filename must end with {suffix}.")
        return name

    def _create_owner_directories(self) -> None:
        activate_session_storage(
            self.settings.session_root,
            owner_storage_key(self.owner),
        )
        self.config_root.mkdir(parents=True, exist_ok=True)
        self.upload_root.mkdir(parents=True, exist_ok=True)

    def acquire_owner_storage(self, *, create_owner: bool = True) -> SessionStorageLease:
        """Lock this owner's storage and refresh it only when it exists."""

        lease = acquire_session_storage_lease(
            self.settings.session_root,
            owner_storage_key(self.owner),
            create_owner=create_owner,
        )
        try:
            if create_owner:
                self.config_root.mkdir(parents=True, exist_ok=True)
                self.upload_root.mkdir(parents=True, exist_ok=True)
        except Exception:
            lease.close()
            raise
        return lease

    def activate_owner_storage(self) -> None:
        """Create owner storage while the current hosted request lock is held."""

        self._create_owner_directories()

    def ensure_owner_dirs(self) -> None:
        """Create owner directories and record activity for direct callers."""

        with self.acquire_owner_storage():
            pass

    def resolve(self, alias: str, *, must_exist: bool = True) -> Path:
        """Resolve a public alias within its authorized managed directory.

        Basename validation and a post-resolution containment check prevent
        traversal and symlink escapes. Paper configurations are read-only by
        convention; session configs and uploads remain owner-scoped.
        """

        raw = str(alias).strip()
        if ":" not in raw:
            raise HostedConfigError("Hosted paths must use a virtual alias.")
        prefix, raw_name = raw.split(":", 1)
        if prefix == "paper-config":
            name = self._safe_name(raw_name, suffix=".json")
            root = self.settings.paper_config_root.resolve(strict=False)
            path = root / name
        elif prefix == "session-config":
            name = self._safe_name(raw_name, suffix=".json")
            root = self.config_root.resolve(strict=False)
            path = root / name
        elif prefix == "upload":
            name = self._safe_name(raw_name)
            root = self.upload_root.resolve(strict=False)
            path = root / name
        else:
            raise HostedConfigError("Unknown hosted path alias.")
        path = path.resolve(strict=False)
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise HostedConfigError("Virtual file escapes its managed directory.") from exc
        if must_exist and not path.is_file():
            raise HostedConfigError("Virtual file does not exist.")
        return path

    def alias_for(self, path: Path | str) -> str:
        """Convert a managed server path to its browser-safe alias."""

        resolved = Path(path).resolve(strict=False)
        for prefix, root in (
            ("paper-config", self.settings.paper_config_root),
            ("session-config", self.config_root),
            ("upload", self.upload_root),
        ):
            try:
                relative = resolved.relative_to(root.resolve(strict=False))
            except ValueError:
                continue
            if len(relative.parts) == 1:
                return f"{prefix}:{relative.name}"
        raise HostedConfigError("Server path has no public alias.")

    def virtual_listing(self, alias: str = ".") -> dict[str, Any]:
        """Return the repository-browser shape for the virtual file tree."""

        normalized = str(alias or ".").strip()
        if normalized in {".", ""}:
            return {
                "path": ".",
                "parent_path": None,
                "entries": [
                    {"name": name, "path": name, "is_dir": True, "is_file": False}
                    for name in self.ROOT_ALIASES
                ],
            }
        if normalized == "configs":
            return {
                "path": "configs",
                "parent_path": ".",
                "entries": [
                    {
                        "name": "paper_runs",
                        "path": "configs/paper_runs",
                        "is_dir": True,
                        "is_file": False,
                    },
                    {
                        "name": "user",
                        "path": "configs/user",
                        "is_dir": True,
                        "is_file": False,
                    },
                ],
            }
        roots = {
            "configs/paper_runs": (self.settings.paper_config_root, "paper-config"),
            "configs/user": (self.config_root, "session-config"),
            "uploads": (self.upload_root, "upload"),
        }
        if normalized not in roots:
            raise HostedConfigError("Unknown virtual directory.")
        root, prefix = roots[normalized]
        entries: list[dict[str, Any]] = []
        if root.is_dir():
            for path in sorted(root.iterdir(), key=lambda item: item.name.lower()):
                if not path.is_file():
                    continue
                if normalized.startswith("configs/") and path.suffix.lower() != ".json":
                    continue
                entries.append(
                    {
                        "name": path.name,
                        "path": f"{prefix}:{path.name}",
                        "is_dir": False,
                        "is_file": True,
                    }
                )
        parent_path = "configs" if normalized.startswith("configs/") else "."
        return {"path": normalized, "parent_path": parent_path, "entries": entries}


def _inside_any(path: Path, roots: Iterable[Path]) -> bool:
    """Check resolved-path containment against authorized input roots."""

    resolved = path.resolve(strict=False)
    for root in roots:
        try:
            resolved.relative_to(root.resolve(strict=False))
            return True
        except ValueError:
            continue
    return False


def _read_gold_table(path: Path) -> tuple[list[str], Iterable[list[str]]]:
    """Open a gold table with extension-aware delimiter fallback.

    The returned row iterator owns the open handle and closes it after normal
    completion or an early consumer exit.
    """

    primary = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    handle = path.open("r", encoding="utf-8-sig", newline="")
    reader = csv.reader(handle, delimiter=primary)
    try:
        header = next(reader)
    except StopIteration:
        handle.close()
        raise HostedConfigError("Gold file is empty.")
    if len(header) <= 1:
        handle.close()
        secondary = "," if primary == "\t" else "\t"
        handle = path.open("r", encoding="utf-8-sig", newline="")
        reader = csv.reader(handle, delimiter=secondary)
        try:
            header = next(reader)
        except StopIteration:
            handle.close()
            raise HostedConfigError("Gold file is empty.")

    def rows() -> Iterable[list[str]]:
        try:
            yield from reader
        finally:
            handle.close()

    return header, rows()


def inspect_gold(config: Mapping[str, Any], limits: HostedLimits) -> dict[str, int]:
    """Validate gold columns, scores, POS policy, tokens, and row limits.

    Counts follow the core loader's strip and optional lowercase semantics so
    admission checks describe the data the model will consume.
    """

    dataset = config.get("dataset")
    if not isinstance(dataset, Mapping):
        raise HostedConfigError("dataset must be an object.")
    path = Path(str(dataset.get("gold", "")))
    header, rows = _read_gold_table(path)
    word_column = str(dataset.get("word_column", "Word"))
    score_column = str(dataset.get("score_column", "Conc.M"))
    pos = dataset.get("pos_filter") if isinstance(dataset.get("pos_filter"), Mapping) else {}
    if not word_column or not score_column or len(word_column) > 100 or len(score_column) > 100:
        raise HostedConfigError("Gold column names must contain 1 to 100 characters.")
    required = [word_column, score_column]
    if bool(pos.get("enabled", False)):
        pos_column = str(pos.get("pos_column", "Dom_Pos"))
        if not pos_column or len(pos_column) > 100:
            raise HostedConfigError("The POS column name must contain 1 to 100 characters.")
        required.append(pos_column)
    missing = [name for name in required if name not in header]
    if missing:
        raise HostedConfigError(f"Gold file is missing required column(s): {missing}")
    index = {name: header.index(name) for name in required}
    lowercase = bool(dataset.get("lowercase", True))
    enabled = bool(pos.get("enabled", False))
    match_mode = str(pos.get("match_mode", "exact")).strip().lower()
    if match_mode not in {"exact", "token_contains"}:
        raise HostedConfigError("POS match mode must be exact or token_contains.")
    token_pattern = str(pos.get("token_pattern", DEFAULT_POS_TOKEN_PATTERN))
    if token_pattern != DEFAULT_POS_TOKEN_PATTERN:
        raise HostedConfigError("Hosted POS tokenization uses the server's fixed separator pattern.")
    raw_tags = pos.get("tags", [])
    if not isinstance(raw_tags, (list, tuple)) or len(raw_tags) > 50:
        raise HostedConfigError("Hosted POS tags must be a list of at most 50 values.")
    tags = {str(tag).strip().lower() for tag in raw_tags}
    if any(len(tag) > limits.max_token_length for tag in tags):
        raise HostedConfigError(
            f"POS tags must not exceed {limits.max_token_length} characters."
        )
    if enabled and not tags:
        raise HostedConfigError("POS tags must not be empty when filtering is enabled.")

    normalized_words: set[str] = set()
    source_rows = 0
    retained_rows = 0
    for row in rows:
        source_rows += 1
        if source_rows > limits.max_gold_rows:
            raise HostedConfigError(
                f"Gold data exceed the {limits.max_gold_rows:,}-row hosted limit."
            )
        if len(row) < len(header):
            row = [*row, *([""] * (len(header) - len(row)))]
        values = {name: row[column_index].strip() for name, column_index in index.items()}
        if any(not values[name] for name in required):
            continue
        if enabled:
            pos_value = values[str(pos.get("pos_column", "Dom_Pos"))].lower()
            matches = (
                pos_value in tags
                if match_mode == "exact"
                else bool(tags.intersection(token for token in re.split(DEFAULT_POS_TOKEN_PATTERN, pos_value) if token))
            )
            if not matches:
                continue
        try:
            score = float(values[score_column])
        except ValueError as exc:
            raise HostedConfigError("Gold scores must be numeric.") from exc
        if not math.isfinite(score):
            raise HostedConfigError("Gold scores must be finite.")
        word = values[word_column].strip()
        word = word.lower() if lowercase else word
        if not word:
            continue
        if len(word) > limits.max_token_length:
            raise HostedConfigError(
                f"Gold token exceeds the {limits.max_token_length}-character limit."
            )
        normalized_words.add(word)
        retained_rows += 1
        if retained_rows > limits.max_gold_rows:
            raise HostedConfigError(
                f"Gold data exceed the {limits.max_gold_rows:,}-row hosted limit."
            )
    if not normalized_words:
        raise HostedConfigError("No usable gold words remain after normalization/filtering.")
    return {
        "source_rows": source_rows,
        "retained_rows": retained_rows,
        "normalized_unique_rows": len(normalized_words),
    }


def inspect_target(path: Path, *, lowercase: bool, limits: HostedLimits) -> dict[str, int]:
    """Validate and count a one-token-per-line prediction vocabulary."""

    source_rows = 0
    retained_rows = 0
    normalized_unique: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            source_rows += 1
            token = line.strip()
            if not token:
                continue
            token = token.lower() if lowercase else token
            if len(token) > limits.max_token_length:
                raise HostedConfigError(
                    f"Target token exceeds the {limits.max_token_length}-character limit."
                )
            retained_rows += 1
            normalized_unique.add(token)
            if retained_rows > limits.max_target_rows:
                raise HostedConfigError(
                    f"Target data exceed the {limits.max_target_rows:,}-row hosted limit."
                )
    if not retained_rows:
        raise HostedConfigError("Target vocabulary has no non-empty rows.")
    return {
        "source_rows": source_rows,
        "retained_rows": retained_rows,
        "normalized_unique_rows": len(normalized_unique),
    }


def _inspect_scored_csv(
    path: Path,
    *,
    word_column: str,
    value_column: str,
    lowercase: bool,
    max_rows: int,
    max_token_length: int,
    source_name: str,
) -> tuple[dict[str, int], set[str]]:
    """Validate a scored CSV and return counts plus normalized word keys.

    The scan enforces finite values, unique normalized words, bounded tokens,
    and a source-specific row ceiling without retaining table rows.
    """

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        missing = [name for name in (word_column, value_column) if name not in fieldnames]
        if missing:
            raise HostedConfigError(
                f"{source_name} file is missing required column(s): {missing}"
            )
        words: set[str] = set()
        row_count = 0
        for row in reader:
            row_count += 1
            if row_count > max_rows:
                raise HostedConfigError(
                    f"{source_name} data exceed the {max_rows:,}-row hosted limit."
                )
            raw_word = row.get(word_column)
            raw_value = row.get(value_column)
            if raw_word is None or not str(raw_word).strip():
                raise HostedConfigError(f"{source_name} words must be non-empty.")
            word = str(raw_word).strip()
            word = word.lower() if lowercase else word
            if len(word) > max_token_length:
                raise HostedConfigError(
                    f"{source_name} token exceeds the {max_token_length}-character limit."
                )
            if word in words:
                raise HostedConfigError(
                    f"Duplicate words found in {source_name} after normalization."
                )
            words.add(word)
            try:
                value = float(str(raw_value).strip())
            except (AttributeError, TypeError, ValueError) as exc:
                raise HostedConfigError(f"{source_name} scores must be numeric.") from exc
            if not math.isfinite(value):
                raise HostedConfigError(f"{source_name} scores must be finite.")
    if not words:
        raise HostedConfigError(f"{source_name} file has no data rows.")
    return {"rows": row_count, "normalized_unique_rows": len(words)}, words


def inspect_holdout_inputs(config: Mapping[str, Any], limits: HostedLimits) -> dict[str, Any]:
    """Validate holdout/prediction tables and require normalized overlap.

    Both tables are scanned under their respective row ceilings with the same
    word normalization used by holdout evaluation. The returned audit records
    table counts and the exact intersection size without retaining rows.
    """

    dataset = config.get("dataset") if isinstance(config.get("dataset"), Mapping) else {}
    prediction = (
        config.get("prediction") if isinstance(config.get("prediction"), Mapping) else {}
    )
    lowercase = bool(dataset.get("lowercase", True))
    holdout_summary, holdout_words = _inspect_scored_csv(
        Path(str(prediction.get("holdout", ""))),
        word_column=str(dataset.get("word_column", "Word")),
        value_column=str(dataset.get("score_column", "Conc.M")),
        lowercase=lowercase,
        max_rows=limits.max_gold_rows,
        max_token_length=limits.max_token_length,
        source_name="holdout",
    )
    prediction_summary, prediction_words = _inspect_scored_csv(
        Path(str(prediction.get("predictions_csv", ""))),
        word_column="word",
        value_column=str(prediction.get("prediction_column", "pred")),
        lowercase=lowercase,
        max_rows=limits.max_target_rows,
        max_token_length=limits.max_token_length,
        source_name="predictions_csv",
    )
    matched = len(holdout_words.intersection(prediction_words))
    if matched == 0:
        raise HostedConfigError("No overlapping words between holdout and prediction files.")
    return {
        "holdout": holdout_summary,
        "predictions": prediction_summary,
        "n_matched": matched,
        "overlap_ratio": matched / max(1, len(holdout_words)),
    }


class HostedPolicy:
    """Canonicalize editor configs into bounded, server-authorized jobs."""

    def __init__(self, settings: HostedSettings) -> None:
        self.settings = settings

    def canonicalize_submission(
        self,
        config: Mapping[str, Any],
        *,
        job_paths: JobPaths,
        allowed_data_roots: Iterable[Path],
        task: str = "prediction-run",
        validate_files: bool = True,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Return a runnable hosted config and preflight audit metadata.

        The transformation replaces client embedding data with one allowlisted
        space, forces job-owned outputs and single-threaded model execution,
        enforces search/report limits, and rejects inputs outside authorized
        storage. Optional file inspection validates the exact task inputs
        before the job enters the durable queue.
        """

        cfg = copy.deepcopy(dict(config))
        limits = self.settings.limits
        dataset = cfg.get("dataset")
        prediction = cfg.get("prediction")
        runtime = cfg.get("runtime")
        embeddings = cfg.get("embeddings")
        if not all(isinstance(item, dict) for item in (dataset, prediction, runtime, embeddings)):
            raise HostedConfigError("Config sections dataset, embeddings, runtime and prediction are required.")

        if task not in {"prediction-run", "prediction-holdout"}:
            raise HostedConfigError("Unknown hosted task.")
        mode = str(embeddings.get("mode", "single"))
        if mode != "single":
            raise HostedConfigError("Hosted runs support exactly one embedding space.")
        active_id = str(embeddings.get("active_space", "")).strip()
        submitted_spaces = [
            item for item in embeddings.get("spaces", []) if isinstance(item, Mapping)
        ]
        active_space = next(
            (item for item in submitted_spaces if str(item.get("id", "")) == active_id),
            submitted_spaces[0] if len(submitted_spaces) == 1 else None,
        )
        public_path = str(active_space.get("path", "")).strip() if active_space else ""
        if public_path.startswith("embedding:"):
            selected_id = public_path.removeprefix("embedding:")
            if not selected_id or "/" in selected_id or "\\" in selected_id:
                raise HostedConfigError("Invalid hosted embedding alias.")
        else:
            selected_id = active_id
        spec = self.settings.embedding_allowlist.get(selected_id)
        embeddings["mode"] = "single"
        embeddings["active_space"] = spec.id
        embeddings["spaces"] = [spec.core_config()]

        runtime["n_jobs"] = 1
        runtime["output_dir"] = str(job_paths.output_dir)
        target_value = prediction.get("target")
        if task == "prediction-run" and target_value not in (None, ""):
            cfg.setdefault("reports", {})["level"] = "full"
            prediction["predictions_csv"] = str(job_paths.output_dir / "vocab_predictions.csv")
        prediction["unmatched_csv"] = str(job_paths.output_dir / "holdout_unmatched.csv")

        k_min = int(prediction.get("k_min", 5))
        k_max = int(prediction.get("k_max", 100))
        k_step = int(prediction.get("k_step", 5))
        if k_min < 1 or k_step < 1 or k_max < k_min or k_max > limits.max_k:
            raise HostedConfigError(f"Hosted k values must be between 1 and {limits.max_k}.")
        candidate_count = ((k_max - k_min) // k_step) + 1
        if candidate_count > limits.max_k_candidates:
            raise HostedConfigError(
                f"Hosted runs permit at most {limits.max_k_candidates} candidate k values."
            )
        if int(prediction.get("cv_folds", 5)) > limits.max_cv_folds:
            raise HostedConfigError(
                f"Hosted runs permit at most {limits.max_cv_folds} CV folds."
            )
        if int(prediction.get("topn_neighbors", 10)) > limits.max_evidence_neighbors:
            raise HostedConfigError(
                f"Hosted reports display at most {limits.max_evidence_neighbors} neighbors."
            )

        roots = tuple(Path(root).resolve(strict=False) for root in allowed_data_roots)
        path_fields: list[tuple[dict[str, Any], str]] = [(dataset, "gold")]
        if task == "prediction-run":
            path_fields.append((prediction, "target"))
        else:
            path_fields.extend(
                [(prediction, "holdout"), (prediction, "predictions_csv")]
            )
        for section, key in path_fields:
            value = section.get(key)
            if value in (None, ""):
                continue
            path = Path(str(value)).resolve(strict=False)
            if not _inside_any(path, roots):
                raise HostedConfigError(f"Hosted input '{key}' is outside session-managed storage.")
            section[key] = str(path)

        audit: dict[str, Any] = {}
        if validate_files:
            gold_path = Path(str(dataset.get("gold", "")))
            if not gold_path.is_file():
                raise HostedConfigError("Gold input does not exist.")
            if task == "prediction-run":
                audit["gold"] = inspect_gold(cfg, limits)
            if task == "prediction-run" and prediction.get("target") not in (None, ""):
                target_path = Path(str(prediction["target"]))
                if not target_path.is_file():
                    raise HostedConfigError("Target input does not exist.")
                audit["target"] = inspect_target(
                    target_path,
                    lowercase=bool(dataset.get("lowercase", True)),
                    limits=limits,
                )
            if task == "prediction-holdout":
                holdout_path = Path(str(prediction.get("holdout", "")))
                predictions_path = Path(str(prediction.get("predictions_csv", "")))
                if not holdout_path.is_file():
                    raise HostedConfigError("Holdout input does not exist.")
                if not predictions_path.is_file():
                    raise HostedConfigError("Prediction input does not exist.")
                audit["holdout"] = inspect_holdout_inputs(cfg, limits)
        return cfg, audit
