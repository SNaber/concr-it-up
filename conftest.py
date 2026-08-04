"""Root pytest path bootstrap for the core package tests."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
CORE_SRC = REPO_ROOT / "packages" / "core" / "src"

repo_root = str(REPO_ROOT.resolve())
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

core_src = str(CORE_SRC.resolve())
if core_src not in sys.path:
    sys.path.insert(0, core_src)
