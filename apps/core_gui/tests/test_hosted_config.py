from __future__ import annotations

import json
from pathlib import Path

import pytest

from apps.core_gui.hosted_config import (
    DEFAULT_POS_TOKEN_PATTERN,
    EmbeddingAllowlist,
    HostedConfigError,
    HostedLimits,
    HostedPathResolver,
    HostedPolicy,
    HostedSettings,
)
from apps.core_gui.job_store import SQLiteJobStore


def _settings(tmp_path: Path, allowlist: EmbeddingAllowlist) -> HostedSettings:
    return HostedSettings(
        repo_root=tmp_path,
        enabled=True,
        access_mode="anonymous",
        secret_key="x" * 32,
        db_path=tmp_path / "queue.sqlite3",
        job_root=tmp_path / "jobs",
        session_root=tmp_path / "sessions",
        paper_config_root=tmp_path / "paper",
        embedding_allowlist=allowlist,
    )


def _config(gold: Path, target: Path, embedding_path: str = "/client/path.bin") -> dict:
    return {
        "dataset": {
            "gold": str(gold),
            "word_column": "word",
            "score_column": "score",
            "lowercase": True,
            "pos_filter": {
                "enabled": False,
                "pos_column": "pos",
                "tags": [],
                "match_mode": "exact",
                "token_pattern": DEFAULT_POS_TOKEN_PATTERN,
            },
        },
        "embeddings": {
            "mode": "single",
            "active_space": "english",
            "spaces": [
                {"id": "english", "kind": "ft_bin", "path": embedding_path, "label": "client"}
            ],
        },
        "model": {"kind": "knn"},
        "runtime": {"output_dir": "/client/output", "n_jobs": -1, "seed": 13},
        "prediction": {
            "test_size": 0.2,
            "cv_folds": 5,
            "k_min": 5,
            "k_max": 100,
            "k_step": 5,
            "topn_neighbors": 20,
            "target": str(target),
            "holdout": None,
            "predictions_csv": "/client/result.csv",
            "unmatched_csv": None,
        },
        "reports": {"level": "core"},
    }


def _allowlist(tmp_path: Path) -> EmbeddingAllowlist:
    embedding = tmp_path / "admin" / "cc.en.300.bin"
    embedding.parent.mkdir(parents=True)
    embedding.write_bytes(b"embedding")
    return EmbeddingAllowlist.from_payload(
        [{"id": "english", "kind": "ft_bin", "path": str(embedding), "label": "English"}],
        base_dir=tmp_path,
    )


def test_hosted_settings_reads_shared_retention_hours(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("CONCRITUP_HOSTED_MODE", "0")
    monkeypatch.setenv("CONCRITUP_RETENTION_HOURS", "12.5")
    monkeypatch.delenv("CONCRITUP_EMBEDDING_ALLOWLIST", raising=False)

    settings = HostedSettings.from_env(repo_root=tmp_path)

    assert settings.limits.retention_hours == 12.5


@pytest.mark.parametrize("value", ["-1", "nan", "inf", "not-a-number"])
def test_hosted_settings_rejects_invalid_retention_hours(
    tmp_path: Path,
    monkeypatch,
    value: str,
) -> None:
    monkeypatch.setenv("CONCRITUP_HOSTED_MODE", "0")
    monkeypatch.setenv("CONCRITUP_RETENTION_HOURS", value)
    monkeypatch.delenv("CONCRITUP_EMBEDDING_ALLOWLIST", raising=False)

    with pytest.raises(HostedConfigError, match="CONCRITUP_RETENTION_HOURS"):
        HostedSettings.from_env(repo_root=tmp_path)


@pytest.mark.parametrize(
    ("relative_root", "message"),
    [
        ("outside", "below the repository data directory"),
        ("data", "dedicated child"),
        ("data/jobs", "overlaps another managed path"),
    ],
)
def test_hosted_settings_confines_session_deletion_root(
    tmp_path: Path,
    monkeypatch,
    relative_root: str,
    message: str,
) -> None:
    monkeypatch.setenv("CONCRITUP_HOSTED_MODE", "0")
    monkeypatch.setenv("CONCRITUP_SESSION_ROOT", str(tmp_path / relative_root))
    monkeypatch.delenv("CONCRITUP_EMBEDDING_ALLOWLIST", raising=False)

    with pytest.raises(HostedConfigError, match=message):
        HostedSettings.from_env(repo_root=tmp_path)


def test_hosted_settings_rejects_final_session_root_symlink(
    tmp_path: Path,
    monkeypatch,
) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    session_root = data_root / "hosted_sessions"
    session_root.symlink_to(outside, target_is_directory=True)
    monkeypatch.setenv("CONCRITUP_HOSTED_MODE", "0")
    monkeypatch.setenv("CONCRITUP_SESSION_ROOT", str(session_root))
    monkeypatch.delenv("CONCRITUP_EMBEDDING_ALLOWLIST", raising=False)

    with pytest.raises(HostedConfigError, match="may not be a symlink"):
        HostedSettings.from_env(repo_root=tmp_path)


def test_canonicalize_forces_server_paths_limits_and_single_job_runtime(tmp_path: Path) -> None:
    inputs = tmp_path / "sessions" / "owner" / "uploads"
    inputs.mkdir(parents=True)
    gold = inputs / "gold.csv"
    target = inputs / "target.txt"
    gold.write_text("word,score\nDog,4.5\nidea,1.4\n", encoding="utf-8")
    target.write_text("Dog\nidea\n", encoding="utf-8")
    settings = _settings(tmp_path, _allowlist(tmp_path))
    store = SQLiteJobStore(settings.db_path, settings.job_root)
    paths = store.paths_for(store.new_job_id())

    canonical, audit = HostedPolicy(settings).canonicalize_submission(
        _config(gold, target),
        job_paths=paths,
        allowed_data_roots=[inputs],
    )

    assert canonical["runtime"] == {
        "output_dir": str(paths.output_dir),
        "n_jobs": 1,
        "seed": 13,
    }
    assert canonical["reports"]["level"] == "full"
    assert canonical["prediction"]["predictions_csv"] == str(
        paths.output_dir / "vocab_predictions.csv"
    )
    assert canonical["embeddings"]["spaces"] == [
        settings.embedding_allowlist.get("english").core_config()
    ]
    assert audit["gold"]["normalized_unique_rows"] == 2
    assert audit["target"]["retained_rows"] == 2


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda cfg: cfg["embeddings"].update(mode="joint"), "exactly one"),
        (lambda cfg: cfg["prediction"].update(k_max=101), "between 1 and 100"),
        (lambda cfg: cfg["prediction"].update(k_min=1, k_max=100, k_step=1), "at most 20"),
        (lambda cfg: cfg["prediction"].update(cv_folds=6), "at most 5"),
        (lambda cfg: cfg["prediction"].update(topn_neighbors=21), "at most 20"),
    ],
)
def test_hosted_model_limits_are_enforced(tmp_path: Path, mutator, message: str) -> None:
    inputs = tmp_path / "uploads"
    inputs.mkdir()
    gold = inputs / "gold.csv"
    target = inputs / "target.txt"
    gold.write_text("word,score\na,1\nb,2\n", encoding="utf-8")
    target.write_text("a\n", encoding="utf-8")
    settings = _settings(tmp_path, _allowlist(tmp_path))
    store = SQLiteJobStore(settings.db_path, settings.job_root)
    cfg = _config(gold, target)
    mutator(cfg)
    with pytest.raises(HostedConfigError, match=message):
        HostedPolicy(settings).canonicalize_submission(
            cfg,
            job_paths=store.paths_for(store.new_job_id()),
            allowed_data_roots=[inputs],
        )


def test_hosted_inputs_cannot_reference_arbitrary_server_paths(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = tmp_path / "private.csv"
    target = allowed / "target.txt"
    outside.write_text("word,score\na,1\n", encoding="utf-8")
    target.write_text("a\n", encoding="utf-8")
    settings = _settings(tmp_path, _allowlist(tmp_path))
    paths = SQLiteJobStore(settings.db_path, settings.job_root).paths_for(
        SQLiteJobStore.new_job_id()
    )
    with pytest.raises(HostedConfigError, match="outside session-managed"):
        HostedPolicy(settings).canonicalize_submission(
            _config(outside, target),
            job_paths=paths,
            allowed_data_roots=[allowed],
        )


def test_nonfinite_gold_and_custom_pos_regex_are_rejected(tmp_path: Path) -> None:
    inputs = tmp_path / "uploads"
    inputs.mkdir()
    gold = inputs / "gold.csv"
    target = inputs / "target.txt"
    gold.write_text("word,score,pos\na,nan,Noun\n", encoding="utf-8")
    target.write_text("a\n", encoding="utf-8")
    settings = _settings(tmp_path, _allowlist(tmp_path))
    paths = SQLiteJobStore(settings.db_path, settings.job_root).paths_for(
        SQLiteJobStore.new_job_id()
    )
    cfg = _config(gold, target)
    cfg["dataset"]["pos_filter"].update(enabled=True, tags=["Noun"], token_pattern=r".*")
    with pytest.raises(HostedConfigError, match="fixed separator"):
        HostedPolicy(settings).canonicalize_submission(
            cfg, job_paths=paths, allowed_data_roots=[inputs]
        )
    cfg["dataset"]["pos_filter"]["token_pattern"] = DEFAULT_POS_TOKEN_PATTERN
    with pytest.raises(HostedConfigError, match="finite"):
        HostedPolicy(settings).canonicalize_submission(
            cfg, job_paths=paths, allowed_data_roots=[inputs]
        )


def test_virtual_paths_are_owner_scoped_and_never_accept_host_paths(tmp_path: Path) -> None:
    settings = _settings(tmp_path, _allowlist(tmp_path))
    one = HostedPathResolver(settings, "owner-one")
    two = HostedPathResolver(settings, "owner-two")
    one.ensure_owner_dirs()
    two.ensure_owner_dirs()
    one_file = one.upload_root / "gold.csv"
    one_file.write_text("word,score\na,1\n", encoding="utf-8")

    assert one.resolve("upload:gold.csv") == one_file.resolve()
    with pytest.raises(HostedConfigError, match="does not exist"):
        two.resolve("upload:gold.csv")
    with pytest.raises(HostedConfigError, match="virtual alias"):
        one.resolve(str(one_file))
    assert one.alias_for(one_file) == "upload:gold.csv"
    assert one.virtual_listing("uploads")["entries"][0]["path"] == "upload:gold.csv"


def test_allowlist_public_shape_hides_absolute_path(tmp_path: Path) -> None:
    allowlist = _allowlist(tmp_path)
    public = allowlist.public_entries()
    assert public[0]["path"] == "embedding:english"
    assert str(tmp_path) not in json.dumps(public)


def test_embedding_path_alias_overrides_stale_editor_space_id(tmp_path: Path) -> None:
    inputs = tmp_path / "uploads"
    inputs.mkdir()
    gold = inputs / "gold.csv"
    target = inputs / "target.txt"
    gold.write_text("word,score\na,1\n", encoding="utf-8")
    target.write_text("a\n", encoding="utf-8")
    allowlist = _allowlist(tmp_path)
    settings = _settings(tmp_path, allowlist)
    paths = SQLiteJobStore(settings.db_path, settings.job_root).paths_for(
        SQLiteJobStore.new_job_id()
    )
    cfg = _config(gold, target)
    cfg["embeddings"]["active_space"] = "stale-id"
    cfg["embeddings"]["spaces"][0]["id"] = "stale-id"
    cfg["embeddings"]["spaces"][0]["path"] = "embedding:english"

    canonical, _ = HostedPolicy(settings).canonicalize_submission(
        cfg, job_paths=paths, allowed_data_roots=[inputs]
    )

    assert canonical["embeddings"]["active_space"] == "english"


def test_holdout_keeps_owner_authorized_predictions_path(tmp_path: Path) -> None:
    inputs = tmp_path / "uploads"
    inputs.mkdir()
    gold = inputs / "gold.csv"
    target = inputs / "target.txt"
    holdout = inputs / "holdout.csv"
    predictions = inputs / "predictions.csv"
    gold.write_text("word,score\na,1\n", encoding="utf-8")
    target.write_text("a\n", encoding="utf-8")
    holdout.write_text("word,score\na,1\n", encoding="utf-8")
    predictions.write_text("word,pred\na,1\n", encoding="utf-8")
    settings = _settings(tmp_path, _allowlist(tmp_path))
    paths = SQLiteJobStore(settings.db_path, settings.job_root).paths_for(
        SQLiteJobStore.new_job_id()
    )
    cfg = _config(gold, target)
    cfg["prediction"]["holdout"] = str(holdout)
    cfg["prediction"]["predictions_csv"] = str(predictions)

    canonical, audit = HostedPolicy(settings).canonicalize_submission(
        cfg,
        job_paths=paths,
        allowed_data_roots=[inputs],
        task="prediction-holdout",
    )

    assert canonical["prediction"]["predictions_csv"] == str(predictions.resolve())
    assert canonical["prediction"]["unmatched_csv"] == str(
        paths.output_dir / "holdout_unmatched.csv"
    )
    assert audit["holdout"]["n_matched"] == 1
    assert audit["holdout"]["overlap_ratio"] == 1.0


def test_gold_source_row_limit_is_not_bypassed_by_duplicate_words(tmp_path: Path) -> None:
    inputs = tmp_path / "uploads"
    inputs.mkdir()
    gold = inputs / "gold.csv"
    target = inputs / "target.txt"
    gold.write_text("word,score\na,1\na,2\n", encoding="utf-8")
    target.write_text("a\n", encoding="utf-8")
    base = _settings(tmp_path, _allowlist(tmp_path))
    settings = HostedSettings(**{**base.__dict__, "limits": HostedLimits(max_gold_rows=1)})
    paths = SQLiteJobStore(settings.db_path, settings.job_root).paths_for(
        SQLiteJobStore.new_job_id()
    )
    with pytest.raises(HostedConfigError, match="1-row hosted limit"):
        HostedPolicy(settings).canonicalize_submission(
            _config(gold, target), job_paths=paths, allowed_data_roots=[inputs]
        )


def test_holdout_rejects_duplicate_words_nonfinite_values_and_no_overlap(tmp_path: Path) -> None:
    inputs = tmp_path / "uploads"
    inputs.mkdir()
    gold = inputs / "gold.csv"
    target = inputs / "target.txt"
    holdout = inputs / "holdout.csv"
    predictions = inputs / "predictions.csv"
    gold.write_text("word,score\na,1\n", encoding="utf-8")
    target.write_text("a\n", encoding="utf-8")
    settings = _settings(tmp_path, _allowlist(tmp_path))
    paths = SQLiteJobStore(settings.db_path, settings.job_root).paths_for(
        SQLiteJobStore.new_job_id()
    )
    cfg = _config(gold, target)
    cfg["prediction"].update(holdout=str(holdout), predictions_csv=str(predictions))

    holdout.write_text("word,score\na,1\nA,2\n", encoding="utf-8")
    predictions.write_text("word,pred\na,1\n", encoding="utf-8")
    with pytest.raises(HostedConfigError, match="Duplicate words"):
        HostedPolicy(settings).canonicalize_submission(
            cfg,
            job_paths=paths,
            allowed_data_roots=[inputs],
            task="prediction-holdout",
        )

    holdout.write_text("word,score\na,nan\n", encoding="utf-8")
    with pytest.raises(HostedConfigError, match="finite"):
        HostedPolicy(settings).canonicalize_submission(
            cfg,
            job_paths=paths,
            allowed_data_roots=[inputs],
            task="prediction-holdout",
        )

    holdout.write_text("word,score\nb,2\n", encoding="utf-8")
    with pytest.raises(HostedConfigError, match="No overlapping"):
        HostedPolicy(settings).canonicalize_submission(
            cfg,
            job_paths=paths,
            allowed_data_roots=[inputs],
            task="prediction-holdout",
        )


def test_virtual_config_browser_preserves_existing_gui_paths(tmp_path: Path) -> None:
    settings = _settings(tmp_path, _allowlist(tmp_path))
    settings.paper_config_root.mkdir(parents=True)
    (settings.paper_config_root / "english.json").write_text("{}", encoding="utf-8")
    resolver = HostedPathResolver(settings, "owner")
    resolver.ensure_owner_dirs()
    (resolver.config_root / "mine.json").write_text("{}", encoding="utf-8")
    assert [entry["path"] for entry in resolver.virtual_listing("configs")["entries"]] == [
        "configs/paper_runs",
        "configs/user",
    ]
    assert resolver.virtual_listing("configs/paper_runs")["entries"][0]["path"] == (
        "paper-config:english.json"
    )
    assert resolver.virtual_listing("configs/user")["entries"][0]["path"] == (
        "session-config:mine.json"
    )
