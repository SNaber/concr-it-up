"""Cross-validation helpers for selecting kNN settings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sklearn.model_selection import KFold

from .knn_backend import create_regressor
from .knn_metrics import rmse, safe_spearman


@dataclass(frozen=True)
class CVResult:
    """Cross-validation summary row for one (`k`, `weights`) setting."""

    k: int
    weights: str
    mean_spearman: float
    std_spearman: float
    mean_rmse: float
    std_rmse: float


def cross_validate_knn(
    x: np.ndarray,
    y: np.ndarray,
    k_values: Iterable[int],
    weights_options: Iterable[str],
    n_splits: int,
    seed: int,
    model_kind: str = "knn",
    n_jobs: int = -1,
) -> list[CVResult]:
    """Evaluate k/weight combinations with deterministic shuffled K-fold CV.

    ``x`` has shape ``(n_samples, n_features)`` and ``y`` has one target per
    row. Candidates too large for a fold's training portion are skipped. Each
    result contains the mean and sample standard deviation of fold Spearman and
    RMSE values.
    """
    if model_kind != "knn":
        raise ValueError("cross_validate_knn currently supports only model.kind='knn'.")

    out: list[CVResult] = []
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for weights in weights_options:
        for k in k_values:
            fold_rhos: list[float] = []
            fold_rmses: list[float] = []
            for train_idx, val_idx in kf.split(x):
                x_train, x_val = x[train_idx], x[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                if k > len(x_train):
                    continue

                model = create_regressor(
                    kind="knn",
                    n_neighbors=int(k),
                    weights=str(weights),
                    n_jobs=int(n_jobs),
                )
                model.fit(x_train, y_train)
                y_hat = model.predict(x_val)
                fold_rhos.append(safe_spearman(y_val, y_hat))
                fold_rmses.append(rmse(y_val, y_hat))

            if not fold_rhos:
                continue

            out.append(
                CVResult(
                    k=int(k),
                    weights=str(weights),
                    mean_spearman=float(np.mean(fold_rhos)),
                    std_spearman=float(np.std(fold_rhos, ddof=1)) if len(fold_rhos) > 1 else 0.0,
                    mean_rmse=float(np.mean(fold_rmses)),
                    std_rmse=float(np.std(fold_rmses, ddof=1)) if len(fold_rmses) > 1 else 0.0,
                )
            )
    return out


def pick_best(results: list[CVResult]) -> CVResult:
    """Select the highest-mean-Spearman row, breaking ties by mean RMSE."""
    if not results:
        raise ValueError("No CV results to select from.")
    return sorted(results, key=lambda row: (-row.mean_spearman, row.mean_rmse))[0]
