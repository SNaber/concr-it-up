#!/usr/bin/env python3
"""Lightweight integrity checks for the repository's public documentation."""

from __future__ import annotations

import re
import shlex
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from concreteness_knn_core.cli import build_parser  # noqa: E402


LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
HEADER_RE = re.compile(r"^#{1,6}\s+(.*?)\s*$")
CODE_FENCE_RE = re.compile(r"```(?:bash|sh|shell)\n(.*?)```", re.DOTALL | re.IGNORECASE)
PUBLIC_MARKDOWN_PATTERNS = (
    "README.md",
    "apps/core_gui/README.md",
    "configs/paper_runs/README.md",
    "docs/*.md",
    "examples/smoke/README.md",
    "packages/core/README.md",
    "packages/core/docs/*.md",
)


def _slugify_heading(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"\s+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text


def _collect_headings(path: Path) -> Set[str]:
    anchors: Set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        m = HEADER_RE.match(line)
        if m:
            anchors.add(_slugify_heading(m.group(1)))
    return anchors


def _split_target(target: str) -> Tuple[str, str]:
    if "#" in target:
        base, anchor = target.split("#", 1)
        return base, anchor
    return target, ""


def _is_external(target: str) -> bool:
    return bool(re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", target))


def check_markdown_links(markdown_files: List[Path]) -> List[str]:
    """Return broken local-link and missing-anchor errors for Markdown files."""
    errors: List[str] = []
    heading_cache: Dict[Path, Set[str]] = {}

    for md in markdown_files:
        text = md.read_text(encoding="utf-8")
        for match in LINK_RE.finditer(text):
            raw_target = match.group(1).strip()
            if not raw_target or _is_external(raw_target):
                continue

            target_path_str, anchor = _split_target(raw_target)
            if target_path_str == "":
                target_path = md
            else:
                target_path = (md.parent / target_path_str).resolve()
            if not target_path.exists():
                errors.append(f"{md}: broken link target '{raw_target}'")
                continue

            if anchor:
                if target_path not in heading_cache:
                    heading_cache[target_path] = _collect_headings(target_path)
                if _slugify_heading(anchor) not in heading_cache[target_path]:
                    errors.append(f"{md}: missing anchor '#{anchor}' in '{target_path}'")
    return errors


def _sanitize_cli_line(line: str) -> List[str] | None:
    """Convert a documented core command into parser tokens when applicable."""
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    if line.endswith("\\"):
        return None
    if "<group>" in line or "<command>" in line:
        return None

    is_core_command = (
        line.startswith("concreteness-knn-core ")
        or line.startswith("PYTHONPATH=")
        or bool(re.match(r"^python(?:3(?:\.\d+)?)?\s+-m\s+concreteness_knn_core\b", line))
    )
    if not is_core_command:
        return None

    if "<file>" in line:
        line = line.replace("<file>", "dummy.json")
    if "<path>" in line:
        line = line.replace("<path>", "dummy-path")
    if "{prediction}" in line:
        line = line.replace("{prediction}", "prediction")

    if line.startswith("PYTHONPATH="):
        parts = shlex.split(line)
        if parts and parts[0].startswith("PYTHONPATH="):
            parts = parts[1:]
    else:
        parts = shlex.split(line)

    if not parts:
        return None

    if parts[0] == "concreteness-knn-core":
        return parts[1:]
    if len(parts) >= 3 and parts[0] == "python" and parts[1] == "-m" and parts[2] == "concreteness_knn_core":
        return parts[3:]
    return None


def check_cli_snippets(markdown_files: List[Path]) -> List[str]:
    """Return errors for documented shell commands rejected by the core parser."""
    parser = build_parser()
    errors: List[str] = []

    for md in markdown_files:
        text = md.read_text(encoding="utf-8")
        for block in CODE_FENCE_RE.findall(text):
            for raw_line in block.splitlines():
                tokens = _sanitize_cli_line(raw_line)
                if tokens is None:
                    continue
                try:
                    parser.parse_args(tokens)
                except SystemExit as exc:
                    if int(exc.code) != 0:
                        errors.append(f"{md}: invalid CLI snippet '{raw_line.strip()}'")
                except Exception as exc:  # pragma: no cover
                    errors.append(f"{md}: failed parsing snippet '{raw_line.strip()}': {exc}")
    return errors


def main() -> int:
    """Check public documentation links and core CLI snippets."""

    markdown_files = sorted(
        {
            path
            for pattern in PUBLIC_MARKDOWN_PATTERNS
            for path in PROJECT_ROOT.glob(pattern)
        }
    )

    errors = []
    errors.extend(check_markdown_links(markdown_files))
    errors.extend(check_cli_snippets(markdown_files))

    if errors:
        for error in errors:
            print(f"[ERROR] {error}")
        return 1

    print("[OK] Documentation links and CLI snippets are valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
