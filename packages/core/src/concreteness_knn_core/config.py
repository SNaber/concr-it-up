"""Core configuration defaults, loading, templating, and validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

SCHEMA_VERSION = "7.0.0"

_LEGACY_KEY_RENAMES: dict[tuple[str, ...], str] = {
    ("dataset", "gold_csv"): "dataset.gold",
    ("dataset", "word_col"): "dataset.word_column",
    ("dataset", "score_col"): "dataset.score_column",
    ("dataset", "pos_filter", "pos_col"): "dataset.pos_filter.pos_column",
    ("embeddings", "single_space_id"): "embeddings.active_space",
    ("runtime", "out_dir"): "runtime.output_dir",
    ("prediction", "predict_vocab"): "prediction.target",
    ("prediction", "holdout_csv"): "prediction.holdout",
    ("prediction", "pred_csv"): "prediction.predictions_csv",
    ("prediction", "pred_col"): "prediction.prediction_column",
    ("prediction", "unmatched_out_csv"): "prediction.unmatched_csv",
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge two dictionaries."""
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def default_config() -> Dict[str, Any]:
    """Return the canonical core configuration defaults."""
    return {
        "dataset": {
            "gold": "",
            "word_column": "Word",
            "score_column": "Conc.M",
            "lowercase": True,
            "pos_filter": {
                "enabled": False,
                "pos_column": "Dom_Pos",
                "tags": ["Noun"],
            },
        },
        "embeddings": {
            "mode": "single",
            "active_space": "default",
            "spaces": [
                {
                    "id": "default",
                    "kind": "ft_bin",
                    "path": "",
                    "label": "default_embedding",
                }
            ],
        },
        "model": {
            "kind": "knn",
        },
        "runtime": {
            "output_dir": "data/generated_scores/und/prediction_run",
            "seed": 13,
            "n_jobs": -1,
        },
        "prediction": {
            "test_size": 0.2,
            "cv_folds": 5,
            "k_min": 5,
            "k_max": 100,
            "k_step": 5,
            "topn_neighbors": 10,
            "target": None,
            "holdout": None,
            "predictions_csv": None,
            "prediction_column": "pred",
            "unmatched_csv": None,
        },
        "reports": {
            "level": "core",
        },
    }


def init_template(task: str) -> Dict[str, Any]:
    """Build a task-oriented starter config from core defaults."""
    cfg = default_config()
    if task == "prediction":
        cfg["dataset"]["gold"] = (
            "data/datasets/en/brysbaert/normalized/Concreteness_ratings_Brysbaert_et_al_BRM_train.csv"
        )
        cfg["embeddings"]["active_space"] = "cc_en_300"
        cfg["embeddings"]["spaces"] = [
            {
                "id": "cc_en_300",
                "kind": "ft_bin",
                "path": "embeddings/cc.en.300.bin",
                "label": "cc_en_300",
            }
        ]
        cfg["runtime"]["output_dir"] = "data/generated_scores/en/prediction_run"
        cfg["prediction"][
            "target"
        ] = (
            "data/datasets/en/brysbaert/normalized/"
            "Concreteness_ratings_Brysbaert_et_al_BRM_holdout_vocab.txt"
        )
        cfg["prediction"]["holdout"] = (
            "data/datasets/en/brysbaert/normalized/Concreteness_ratings_Brysbaert_et_al_BRM_holdout.csv"
        )
        cfg["prediction"]["predictions_csv"] = "data/generated_scores/en/prediction_run/vocab_predictions.csv"
        cfg["prediction"]["unmatched_csv"] = "data/generated_scores/en/prediction_run/holdout_unmatched.csv"
        return cfg

    raise ValueError("task must be: prediction")


def load_config(config_path: str) -> Dict[str, Any]:
    """Load JSON config and merge it on top of core defaults."""
    payload = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Config file must contain a JSON object.")
    return _deep_merge(default_config(), payload)


def space_index(embeddings_cfg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Return `space_id -> space config` mapping."""
    spaces = embeddings_cfg.get("spaces", [])
    return {str(space["id"]): dict(space) for space in spaces if isinstance(space, dict)}


def active_space_ids(embeddings_cfg: Dict[str, Any]) -> list[str]:
    """Return active embedding-space ids for the current embedding mode."""
    mode = str(embeddings_cfg.get("mode", "single"))
    space_ids = [
        str(space["id"]) for space in embeddings_cfg.get("spaces", []) if isinstance(space, dict)
    ]
    if mode == "single":
        single_id = embeddings_cfg.get("active_space")
        return [str(single_id)] if single_id is not None else []
    if mode in {"joint", "multi_space"}:
        return space_ids
    return []


def write_template(task: str, out_path: str) -> Path:
    """Write a starter JSON template to disk."""
    cfg = init_template(task)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return out


def _legacy_keys_present(cfg: Dict[str, Any]) -> list[tuple[str, str]]:
    present: list[tuple[str, str]] = []
    for old_parts, new_path in _LEGACY_KEY_RENAMES.items():
        node: Any = cfg
        for part in old_parts[:-1]:
            if not isinstance(node, dict):
                node = None
                break
            node = node.get(part)
        if isinstance(node, dict) and old_parts[-1] in node:
            present.append((".".join(old_parts), new_path))
    return present


def validate_common_config(cfg: Dict[str, Any]) -> None:
    """Validate cross-task invariants shared by core and extension workflows."""
    legacy_pairs = _legacy_keys_present(cfg)
    if legacy_pairs:
        details = "; ".join(f"{old} -> {new}" for old, new in legacy_pairs)
        raise ValueError(f"Legacy config keys are not supported: {details}")

    dataset = cfg["dataset"]
    embeddings = cfg["embeddings"]
    model = cfg["model"]
    runtime = cfg["runtime"]
    reports = cfg["reports"]

    if not dataset["gold"]:
        raise ValueError("dataset.gold must be provided.")
    if model["kind"] != "knn":
        raise ValueError("model.kind must be 'knn'.")
    if int(runtime["n_jobs"]) == 0:
        raise ValueError("runtime.n_jobs must be != 0.")
    if reports["level"] not in {"core", "full"}:
        raise ValueError("reports.level must be 'core' or 'full'.")

    if not isinstance(embeddings, dict):
        raise ValueError("embeddings must be an object.")
    mode = str(embeddings.get("mode", "single"))
    if mode not in {"single", "joint", "multi_space"}:
        raise ValueError("embeddings.mode must be one of: single, joint, multi_space.")

    spaces = embeddings.get("spaces")
    if not isinstance(spaces, list) or not spaces:
        raise ValueError("embeddings.spaces must be a non-empty list.")

    seen_ids = set()
    for idx, space in enumerate(spaces):
        if not isinstance(space, dict):
            raise ValueError(f"embeddings.spaces[{idx}] must be an object.")
        space_id = str(space.get("id", "")).strip()
        if not space_id:
            raise ValueError(f"embeddings.spaces[{idx}].id must be a non-empty string.")
        if space_id in seen_ids:
            raise ValueError("embeddings.spaces ids must be unique.")
        seen_ids.add(space_id)
        if str(space.get("kind", "")) not in {"ft_bin", "vec"}:
            raise ValueError(f"embeddings.spaces[{idx}].kind must be 'ft_bin' or 'vec'.")

    if mode == "single":
        single_id = str(embeddings.get("active_space", "")).strip()
        if not single_id:
            raise ValueError("embeddings.active_space must be provided for mode='single'.")
        if single_id not in seen_ids:
            raise ValueError("embeddings.active_space must match one of embeddings.spaces[*].id.")
    else:
        if len(spaces) < 2:
            raise ValueError("embeddings.mode='joint' or 'multi_space' requires at least 2 spaces.")

    pos_filter = dataset["pos_filter"]
    if bool(pos_filter["enabled"]) and not pos_filter["tags"]:
        raise ValueError("dataset.pos_filter.tags must not be empty when enabled=true.")


def validate_config(cfg: Dict[str, Any], task: str) -> None:
    """Validate core config for prediction commands."""
    validate_common_config(cfg)

    if task == "prediction_run":
        prediction_cfg = cfg["prediction"]
        embeddings_cfg = cfg["embeddings"]
        spaces_by_id = space_index(embeddings_cfg)
        for space_id in active_space_ids(embeddings_cfg):
            if not str(spaces_by_id.get(space_id, {}).get("path", "")).strip():
                raise ValueError(
                    f"embeddings.spaces entry for id='{space_id}' must provide a non-empty path."
                )
        if not (0.0 < float(prediction_cfg["test_size"]) < 1.0):
            raise ValueError("prediction.test_size must be in (0, 1).")
        if int(prediction_cfg["cv_folds"]) < 2:
            raise ValueError("prediction.cv_folds must be >= 2.")
        if int(prediction_cfg["k_step"]) < 1:
            raise ValueError("prediction.k_step must be >= 1.")
        if int(prediction_cfg["k_min"]) < 1 or int(prediction_cfg["k_max"]) < 1:
            raise ValueError("prediction.k_min and prediction.k_max must be >= 1.")
        if int(prediction_cfg["k_min"]) > int(prediction_cfg["k_max"]):
            raise ValueError("prediction.k_min must be <= prediction.k_max.")
        if int(prediction_cfg["topn_neighbors"]) < 1:
            raise ValueError("prediction.topn_neighbors must be >= 1.")
        return

    if task == "prediction_holdout":
        prediction_cfg = cfg["prediction"]
        if not prediction_cfg["holdout"]:
            raise ValueError("prediction.holdout must be provided for holdout scoring.")
        if not prediction_cfg["predictions_csv"]:
            raise ValueError("prediction.predictions_csv must be provided for holdout scoring.")
        return

    raise ValueError(f"Unknown core validation task: {task}")
