"""Metrics used by model selection and held-out prediction evaluation."""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error


def safe_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Spearman rho with deterministic NaN handling."""
    rho, _ = spearmanr(y_true, y_pred)
    if np.isnan(rho):
        return 0.0
    return float(rho)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute root mean squared error."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))
