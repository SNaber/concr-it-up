"""kNN backend and runtime helpers."""

from __future__ import annotations

import os
from typing import Protocol, Tuple

import numpy as np
from sklearn.neighbors import KNeighborsRegressor

from .knn_subset_eval import topk_indices_and_distances


class RegressorBackend(Protocol):
    """Regression interface required by the prediction trainer."""

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> "RegressorBackend":
        """Fit on ``(n_samples, n_features)`` vectors and ``(n_samples,)`` targets."""
        ...

    def predict(self, x_target: np.ndarray) -> np.ndarray:
        """Predict one score per row of an ``(n_targets, n_features)`` matrix."""
        ...

    def predict_with_neighbors(
        self,
        x_target: np.ndarray,
        topn: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return predictions plus neighbor indices and distances for each target.

        The two evidence arrays have shape ``(n_targets, min(topn, n_train))``.
        Indices address the fitted training rows; distance interpretation and
        ordering of exactly tied neighbors are backend-specific.
        """
        ...


def resolve_n_jobs(n_jobs: int) -> int:
    """Resolve `n_jobs` semantics against available CPUs."""
    cpu_count = os.cpu_count() or 1
    if n_jobs == -1:
        return cpu_count
    if n_jobs < -1:
        return max(1, cpu_count + 1 + n_jobs)
    if n_jobs == 0:
        raise ValueError("n_jobs must be != 0.")
    return int(n_jobs)


class KNNBackend:
    """Brute-force cosine-distance kNN regression and neighbor evidence.

    ``predict`` delegates score calculation to scikit-learn. Neighbor evidence
    is computed from the retained float32 training matrix and uses cosine
    distance ``1 - cosine_similarity``.
    """

    def __init__(self, n_neighbors: int, weights: str, n_jobs: int):
        self._model = KNeighborsRegressor(
            n_neighbors=int(n_neighbors),
            metric="cosine",
            weights=str(weights),
            algorithm="brute",
            n_jobs=int(n_jobs),
        )
        self._x_train: np.ndarray | None = None

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> "KNNBackend":
        """Fit the regressor and retain float32 training vectors for evidence."""
        self._model.fit(x_train, y_train)
        self._x_train = np.asarray(x_train, dtype=np.float32)
        return self

    def predict(self, x_target: np.ndarray) -> np.ndarray:
        """Return float32 predictions for the rows in ``x_target``.

        ``fit`` must be called first; scikit-learn raises its fitted-state error
        otherwise.
        """
        return np.asarray(self._model.predict(x_target), dtype=np.float32)

    def predict_with_neighbors(
        self,
        x_target: np.ndarray,
        topn: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict scores and return exact cosine-neighbor evidence.

        The returned tuple contains ``(predictions, indices, distances)``.
        Evidence arrays have one row per target and at most ``topn`` columns;
        indices address rows supplied to ``fit``. Equal cosine similarities do
        not receive a lexical tie-break, so their relative order is unspecified.

        Raises:
            ValueError: If the backend has not been fitted.
        """
        if self._x_train is None:
            raise ValueError("Model must be fit before predict_with_neighbors.")
        preds = self.predict(x_target)
        topk_idx, topk_dist = topk_indices_and_distances(
            train_vectors=self._x_train,
            target_vectors=x_target,
            k=min(max(1, int(topn)), len(self._x_train)),
        )
        return preds, topk_idx, topk_dist


def create_regressor(kind: str, **kwargs) -> RegressorBackend:
    """Regressor backend factory (currently only `knn`)."""
    if kind != "knn":
        raise ValueError(f"Unsupported model kind: {kind}")
    return KNNBackend(
        n_neighbors=int(kwargs["n_neighbors"]),
        weights=str(kwargs["weights"]),
        n_jobs=int(kwargs["n_jobs"]),
    )
