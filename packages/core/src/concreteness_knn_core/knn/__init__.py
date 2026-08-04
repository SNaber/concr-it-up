"""kNN package split into backend, metrics, selection, and subset evaluation."""

from .knn_backend import KNNBackend, RegressorBackend, create_regressor, resolve_n_jobs
from .knn_metrics import rmse, safe_spearman
from .knn_selection import CVResult, cross_validate_knn, pick_best
from .knn_subset_eval import (
    evaluate_candidate,
    normalize_rows,
    predict_with_subset,
    resolve_k,
    topk_indices_and_distances,
)

__all__ = [
    "CVResult",
    "KNNBackend",
    "RegressorBackend",
    "create_regressor",
    "cross_validate_knn",
    "evaluate_candidate",
    "normalize_rows",
    "pick_best",
    "predict_with_subset",
    "resolve_k",
    "resolve_n_jobs",
    "rmse",
    "safe_spearman",
    "topk_indices_and_distances",
]
