"""Exact cosine-neighbor and support-subset utilities."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .knn_metrics import rmse, safe_spearman


_TOPK_TARGET_BATCH_SIZE = 512


def normalize_rows(x: np.ndarray) -> np.ndarray:
    """L2-normalize matrix rows while leaving zero rows finite."""
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(denom, 1e-12, None)


def topk_indices_and_distances(
    train_vectors: np.ndarray,
    target_vectors: np.ndarray,
    k: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return exact top-k cosine-neighbor indices and distances.

    ``train_vectors`` and ``target_vectors`` must be two-dimensional with the
    same feature width. Results have shape
    ``(n_targets, min(k, n_reference))`` and retain target-row order. Indices
    address ``train_vectors``; distances are ``1 - cosine_similarity``. Equal
    similarities have no secondary tie-break, so their relative order follows
    NumPy's partition/sort behavior.
    """
    return _topk_indices_and_distances_batched(
        train_vectors=train_vectors,
        target_vectors=target_vectors,
        k=k,
        batch_size=_TOPK_TARGET_BATCH_SIZE,
    )


def _topk_indices_and_distances_batched(
    train_vectors: np.ndarray,
    target_vectors: np.ndarray,
    k: int,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute exact cosine neighbors with bounded target-side working memory.

    The reference matrix is normalized once. ``batch_size`` bounds only the
    temporary similarity and partition matrices; final result arrays retain the
    full target order.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0.")

    train_norm = normalize_rows(train_vectors)
    n_targets = int(target_vectors.shape[0])
    n_reference = int(train_norm.shape[0])
    result_width = min(int(k), n_reference)

    # Infer result dtypes without allocating a target-by-reference matrix.
    # Normalization retains floating input precision and promotes integers to
    # float64; NumPy neighbor indices use the platform integer dtype.
    index_result = np.empty((n_targets, result_width), dtype=np.intp)
    empty_sim = normalize_rows(target_vectors[:0]) @ train_norm.T
    distance_dtype = empty_sim.dtype
    distance_result = np.empty((n_targets, result_width), dtype=distance_dtype)

    for start in range(0, n_targets, int(batch_size)):
        stop = min(start + int(batch_size), n_targets)
        target_norm = normalize_rows(target_vectors[start:stop])
        sim = target_norm @ train_norm.T

        if k >= sim.shape[1]:
            topk_idx = np.argsort(-sim, axis=1)[:, :k]
        else:
            part = np.argpartition(-sim, k - 1, axis=1)[:, :k]
            part_sim = np.take_along_axis(sim, part, axis=1)
            order = np.argsort(-part_sim, axis=1)
            topk_idx = np.take_along_axis(part, order, axis=1)

        topk_sim = np.take_along_axis(sim, topk_idx, axis=1)
        index_result[start:stop] = topk_idx
        distance_result[start:stop] = 1.0 - topk_sim

    return index_result, distance_result


def resolve_k(k: Optional[int], k_ratio: Optional[float], subset_size: int) -> int:
    """Resolve effective neighbor count from absolute or ratio mode."""
    if subset_size <= 0:
        raise ValueError("subset_size must be > 0.")
    if k_ratio is not None:
        if k_ratio <= 0:
            raise ValueError("k_ratio must be > 0.")
        k_eff = max(1, int(round(k_ratio * subset_size)))
    elif k is not None:
        if k <= 0:
            raise ValueError("k must be > 0.")
        k_eff = int(k)
    else:
        raise ValueError("Either k or k_ratio must be provided.")
    return min(k_eff, subset_size)


def predict_with_subset(
    train_vectors: np.ndarray,
    train_scores: np.ndarray,
    target_vectors: np.ndarray,
    subset_indices: Optional[np.ndarray] = None,
    k: Optional[int] = None,
    k_ratio: Optional[float] = None,
    weighting: str = "distance",
) -> np.ndarray:
    """Predict scores from an optional subset of reference rows.

    ``subset_indices`` addresses ``train_vectors`` and ``train_scores``. The
    effective neighbor count is either the absolute ``k`` or ``k_ratio`` times
    the subset size, capped at that size. Distance weighting uses reciprocal
    cosine distance with a numerical floor of ``1e-12``.
    """
    if subset_indices is None:
        subset_indices = np.arange(len(train_scores), dtype=np.int32)
    else:
        subset_indices = np.asarray(subset_indices, dtype=np.int32)

    if subset_indices.size == 0:
        raise ValueError("subset_indices must contain at least one item.")
    if np.any(subset_indices < 0) or np.any(subset_indices >= len(train_scores)):
        raise ValueError("subset_indices contains out-of-range indices.")

    allowed_vectors = train_vectors[subset_indices]
    allowed_scores = train_scores[subset_indices]
    k_eff = resolve_k(k=k, k_ratio=k_ratio, subset_size=len(subset_indices))

    topk_idx, topk_dist = topk_indices_and_distances(allowed_vectors, target_vectors, k=k_eff)
    neighbor_scores = allowed_scores[topk_idx]

    if weighting == "uniform":
        return neighbor_scores.mean(axis=1)
    if weighting != "distance":
        raise ValueError("weighting must be 'uniform' or 'distance'.")

    weights = 1.0 / np.clip(topk_dist, 1e-12, None)
    return (weights * neighbor_scores).sum(axis=1) / np.clip(weights.sum(axis=1), 1e-12, None)


def evaluate_candidate(
    train_vectors: np.ndarray,
    train_scores: np.ndarray,
    target_vectors: np.ndarray,
    target_scores: np.ndarray,
    subset_indices: Optional[np.ndarray] = None,
    k: Optional[int] = None,
    k_ratio: Optional[float] = None,
    weighting: str = "distance",
) -> dict:
    """Return Spearman, RMSE, and predictions for one support-subset setting."""
    preds = predict_with_subset(
        train_vectors=train_vectors,
        train_scores=train_scores,
        target_vectors=target_vectors,
        subset_indices=subset_indices,
        k=k,
        k_ratio=k_ratio,
        weighting=weighting,
    )
    return {
        "spearman": safe_spearman(target_scores, preds),
        "rmse": rmse(target_scores, preds),
        "predictions": preds,
    }
