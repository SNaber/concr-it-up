"""Typed payloads passed between prediction pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from concreteness_knn_core.data import Embeddings
from concreteness_knn_core.knn import RegressorBackend


@dataclass
class PredictionPreparedData:
    """Gold data, embedding coverage, and active view matrices."""

    embedding_mode: str
    configured_space_ids: list[str]
    view_ids: list[str]
    words_all: list[str]
    y_all: np.ndarray
    words_kept: list[str]
    y_kept: np.ndarray
    dropped_gold_words: list[str]
    coverage: float
    embeddings_by_space: dict[str, Embeddings]
    x_all_by_view: dict[str, np.ndarray]


@dataclass
class PredictionSplitData:
    """Outer train/test split with per-view feature matrices."""

    train_idx: np.ndarray
    test_idx: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    w_train: list[str]
    w_test: list[str]
    x_train_by_view: dict[str, np.ndarray]
    x_test_by_view: dict[str, np.ndarray]
    view_ids: list[str]


@dataclass
class PredictionTrainingResult:
    """CV choice, fitted per-view models, and test/vocab predictions."""

    cv_rows: list[dict[str, Any]]
    best_params: dict[str, Any]
    n_jobs: int
    models_by_view: dict[str, RegressorBackend]
    test_predictions_by_view: dict[str, np.ndarray]
    test_metrics_by_view: dict[str, dict[str, float]]
    merged_test_predictions: np.ndarray
    test_spearman: float
    test_rmse: float
    vocab_rows: list[dict[str, Any]] | None = None
    dropped_vocab_words: list[str] | None = None
    vocab_self_filter_stats: dict[str, Any] | None = None
    vocab_fit_scope: str | None = None
    n_vocab_reference_words: int | None = None


@dataclass
class PredictionHoldoutResult:
    """Holdout scoring output used by report writing."""

    summary_payload: dict[str, Any]
    holdout_df: pd.DataFrame
    pred_df: pd.DataFrame
    merged_df: pd.DataFrame
