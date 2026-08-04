#!/usr/bin/env python3
"""Generate deterministic reference documentation tables for core package."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from concreteness_knn_core.config import default_config  # noqa: E402


GENERATED_CONFIG = REPO_ROOT / "docs" / "generated_config_reference.md"
GENERATED_OUTPUTS = REPO_ROOT / "docs" / "generated_output_manifest.md"


RANGE_NOTES = {
    "runtime.n_jobs": "nonzero",
    "reports.level": "core|full",
    "embeddings.mode": "single|joint|multi_space",
    "embeddings.active_space": "required in single mode; member of spaces[*].id",
    "embeddings.spaces": "non-empty list; unique ids; each kind in {ft_bin, vec}",
    "model.kind": "knn",
    "prediction.test_size": "(0,1)",
    "prediction.cv_folds": ">=2",
    "prediction.k_step": ">=1",
    "prediction.k_min": ">=1",
    "prediction.k_max": ">=1 and >= k_min",
    "prediction.topn_neighbors": ">=1",
}


OUTPUT_MANIFEST: Dict[str, Dict[str, list[str]]] = {
    "prediction_run": {
        "core": [
            "summary.json",
            "cv_results.csv",
            "test_predictions.csv",
        ],
        "full_additions": [
            "oov_gold.csv",
            "vocab_predictions.csv (when prediction.target is set)",
            "oov_vocab.csv",
        ],
    },
    "prediction_holdout": {
        "core": ["holdout_summary.json"],
        "full_additions": ["prediction.unmatched_csv (when configured)"],
    },
}


def _flatten(prefix: str, value: Any) -> Iterable[Tuple[str, Any]]:
    if isinstance(value, dict):
        for key in sorted(value.keys()):
            next_prefix = f"{prefix}.{key}" if prefix else key
            yield from _flatten(next_prefix, value[key])
    else:
        yield prefix, value


def _type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "dict"
    return type(value).__name__


def _format_default(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def build_config_reference() -> str:
    """Render the default prediction configuration as a Markdown table."""

    cfg = default_config()
    rows = list(_flatten("", cfg))
    lines = [
        "# Generated Config Reference",
        "",
        "Auto-generated from `default_config()` in `src/concreteness_knn_core/config.py`.",
        "",
        "| Key | Type | Default | Valid/Notes |",
        "| --- | --- | --- | --- |",
    ]
    for key, value in rows:
        default_repr = _format_default(value).replace("|", "\\|")
        note = RANGE_NOTES.get(key, "")
        lines.append(f"| `{key}` | `{_type_name(value)}` | `{default_repr}` | {note} |")
    lines.append("")
    return "\n".join(lines)


def build_output_manifest() -> str:
    """Render the documented output contract as Markdown."""

    lines = [
        "# Generated Output Manifest",
        "",
        "Auto-generated from `tools/generate_reference_docs.py` contract tables.",
        "",
    ]
    for task in sorted(OUTPUT_MANIFEST.keys()):
        lines.append(f"## `{task}`")
        lines.append("")
        lines.append("Core:")
        lines.append("")
        for item in OUTPUT_MANIFEST[task]["core"]:
            lines.append(f"- `{item}`")
        lines.append("")
        lines.append("Full additions:")
        lines.append("")
        for item in OUTPUT_MANIFEST[task]["full_additions"]:
            lines.append(f"- `{item}`")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _check_or_write(path: Path, content: str, check: bool) -> bool:
    if check:
        if not path.exists():
            print(f"[ERROR] Missing generated file: {path}")
            return False
        existing = path.read_text(encoding="utf-8")
        if existing != content:
            print(f"[ERROR] Out-of-date generated file: {path}")
            return False
        print(f"[OK] Up-to-date: {path}")
        return True

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    print(f"[INFO] Wrote {path}")
    return True


def main() -> int:
    """Generate reference files or verify that tracked copies are current."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Fail if generated docs are out of date.")
    args = parser.parse_args()

    config_md = build_config_reference()
    outputs_md = build_output_manifest()
    ok1 = _check_or_write(GENERATED_CONFIG, config_md, check=args.check)
    ok2 = _check_or_write(GENERATED_OUTPUTS, outputs_md, check=args.check)
    return 0 if (ok1 and ok2) else 1


if __name__ == "__main__":
    raise SystemExit(main())
