"""Web interface for ``concreteness-knn-core``."""

from __future__ import annotations

from typing import Any

__all__ = ["create_app"]


def __getattr__(name: str) -> Any:
    """Expose ``create_app`` without importing Flask during worker startup."""

    if name == "create_app":
        from .app import create_app

        return create_app
    raise AttributeError(name)
