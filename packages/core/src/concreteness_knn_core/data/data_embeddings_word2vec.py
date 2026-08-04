"""word2vec text `.vec` embedding adapter."""

from __future__ import annotations

from typing import Optional

import numpy as np

from .data_embeddings_protocol import Embeddings


class Word2VecTextEmbeddings(Embeddings):
    """Embedding adapter for word2vec text `.vec` files."""

    def __init__(self, vec_path: str, limit: int | None = None):
        """Load vectors from a text-format word2vec file."""
        from gensim.models import KeyedVectors

        self._kv = KeyedVectors.load_word2vec_format(vec_path, binary=False, limit=limit)
        self.dim = int(self._kv.vector_size)

    def get_vector(self, word: str) -> Optional[np.ndarray]:
        """Return a vector when token exists in the loaded vocabulary."""
        if word in self._kv:
            return np.asarray(self._kv[word], dtype=np.float32)
        return None
