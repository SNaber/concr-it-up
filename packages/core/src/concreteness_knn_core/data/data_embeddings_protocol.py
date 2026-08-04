"""Embedding adapter protocol used by vectorization and pipelines."""

from __future__ import annotations

from typing import Optional

import numpy as np


class Embeddings:
    """Abstract embedding adapter interface."""

    dim: int

    def get_vector(self, word: str) -> Optional[np.ndarray]:
        """Return a vector for `word` or `None` if unavailable."""
        raise NotImplementedError
