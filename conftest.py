"""Root pytest path bootstrap for the core package tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent
CORE_SRC = REPO_ROOT / "packages" / "core" / "src"

repo_root = str(REPO_ROOT.resolve())
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

core_src = str(CORE_SRC.resolve())
if core_src not in sys.path:
    sys.path.insert(0, core_src)


@pytest.fixture(autouse=True)
def stable_job_storage_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep tests independent of the host filesystem's existing occupancy."""

    from apps.core_gui.job_store import SQLiteJobStore

    monkeypatch.setattr(
        SQLiteJobStore,
        "storage_usage",
        lambda _store: {
            "total_bytes": 100,
            "available_bytes": 50,
            "used_bytes": 50,
            "used_fraction": 0.5,
        },
    )
