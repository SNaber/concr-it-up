"""Vectorization helpers for single-space, joint, and multi-space modes."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from .data_embeddings_protocol import Embeddings


def vectorize_words(words: Sequence[str], embeddings: Embeddings) -> Tuple[np.ndarray, List[int]]:
    """Convert words to vectors, retaining indices with available embeddings."""
    vectors: List[np.ndarray] = []
    kept_idx: List[int] = []
    for idx, word in enumerate(words):
        vector = embeddings.get_vector(word)
        if vector is None:
            continue
        vectors.append(np.asarray(vector, dtype=np.float32))
        kept_idx.append(int(idx))
    if not vectors:
        raise ValueError("No words had embeddings. Check normalization and embedding file.")
    return np.vstack(vectors), kept_idx


def vectorize_words_intersection(
    words: Sequence[str],
    embeddings_by_space: Dict[str, Embeddings],
) -> Tuple[Dict[str, np.ndarray], List[int]]:
    """Vectorize words using strict intersection across embedding spaces."""
    if not embeddings_by_space:
        raise ValueError("embeddings_by_space must not be empty.")

    vectors_by_space: Dict[str, List[np.ndarray]] = {space_id: [] for space_id in embeddings_by_space}
    kept_idx: List[int] = []

    for idx, word in enumerate(words):
        row_vectors: Dict[str, np.ndarray] = {}
        missing = False
        for space_id, embeddings in embeddings_by_space.items():
            vector = embeddings.get_vector(word)
            if vector is None:
                missing = True
                break
            row_vectors[space_id] = np.asarray(vector, dtype=np.float32)
        if missing:
            continue

        kept_idx.append(int(idx))
        for space_id, vector in row_vectors.items():
            vectors_by_space[space_id].append(vector)

    if not kept_idx:
        raise ValueError("No words had embeddings in the intersection of all spaces.")

    return {space_id: np.vstack(rows) for space_id, rows in vectors_by_space.items()}, kept_idx


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    """Normalize rows for joint features while keeping zero vectors finite."""
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(denom, 1e-12, None)


def build_joint_vectors(vectors_by_space: Dict[str, np.ndarray]) -> np.ndarray:
    """Build joint vectors by per-space row normalization then concatenation."""
    if not vectors_by_space:
        raise ValueError("vectors_by_space must not be empty.")

    ordered_arrays = list(vectors_by_space.values())
    n_rows = int(ordered_arrays[0].shape[0])
    if any(int(arr.shape[0]) != n_rows for arr in ordered_arrays):
        raise ValueError("All embedding spaces must have the same number of rows for joint vectors.")

    normalized = [_normalize_rows(np.asarray(arr, dtype=np.float32)) for arr in ordered_arrays]
    return np.hstack(normalized).astype(np.float32, copy=False)
