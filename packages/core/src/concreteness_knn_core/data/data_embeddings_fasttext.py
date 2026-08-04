"""FastText `.bin` embedding adapter."""

from __future__ import annotations

from typing import Optional

import numpy as np

from .data_embeddings_protocol import Embeddings


class FastTextBinEmbeddings(Embeddings):
    """Embedding adapter for FastText `.bin` models."""

    def __init__(self, bin_path: str):
        """Load a FastText model from disk."""
        import fasttext

        self._model = fasttext.load_model(bin_path)
        self.dim = len(self._model.get_word_vector("test"))

    def get_vector(self, word: str) -> Optional[np.ndarray]:
        """Return the FastText vector for a token."""
        return np.asarray(self._model.get_word_vector(word), dtype=np.float32)
