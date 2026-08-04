"""Prediction model selection, final training, and optional vocab prediction."""

from __future__ import annotations

import math

import numpy as np
from sklearn.model_selection import KFold

from concreteness_knn_core.knn import (
    RegressorBackend,
    create_regressor,
    cross_validate_knn,
    resolve_n_jobs,
    rmse,
    safe_spearman,
)

from .prediction_types import (
    PredictionPreparedData,
    PredictionSplitData,
    PredictionTrainingResult,
)


class PredictionTrainer:
    """Select kNN settings, evaluate the outer split, and predict targets.

    Test metrics come from models fitted only on the outer training portion.
    If target features are supplied, separate models with the selected settings
    are fitted on all covered gold items after evaluation.
    """

    @staticmethod
    def _mean(values: list[float]) -> float:
        return float(np.mean(values)) if values else float("nan")

    @staticmethod
    def _weighted_knn_prediction(scores: np.ndarray, distances: np.ndarray, weights: str) -> float:
        """Compute kNN prediction from neighbor scores/distances."""
        if scores.size == 0:
            raise ValueError("Cannot compute prediction with zero neighbors.")
        if weights == "uniform":
            return float(np.mean(scores))
        if weights == "distance":
            w = 1.0 / np.clip(distances, 1e-12, None)
            return float((w * scores).sum() / np.clip(w.sum(), 1e-12, None))
        raise ValueError(f"Unsupported weights: {weights}")

    @staticmethod
    def _k_values(n_train: int, cv_folds: int, prediction_cfg: dict) -> list[int]:
        """Return configured k values that fit inside every CV training fold."""
        if n_train < 2:
            raise ValueError("Need at least 2 training samples for cross-validation.")
        folds = min(int(cv_folds), n_train)
        if folds < 2:
            raise ValueError("Cross-validation requires at least 2 folds.")

        min_train_size = n_train - math.ceil(n_train / folds)
        if min_train_size < 1:
            raise ValueError("Not enough samples per fold for kNN training.")

        values = [
            int(k)
            for k in range(
                int(prediction_cfg["k_min"]),
                int(prediction_cfg["k_max"]) + 1,
                int(prediction_cfg["k_step"]),
            )
            if k <= min_train_size
        ]
        if not values:
            raise ValueError("No valid k values for this split. Adjust prediction.k_* settings.")
        return values

    def _multi_space_cv_rows(
        self,
        split: PredictionSplitData,
        seed: int,
        cv_folds: int,
        k_values: list[int],
        model_kind: str,
        n_jobs: int,
    ) -> list[dict]:
        """Evaluate each k/weight pair independently in every active space.

        Fold metrics are averaged across spaces for parameter selection while
        per-space means and standard deviations remain available for reports.
        """
        rows: list[dict] = []
        n_train = len(split.y_train)
        folds = min(int(cv_folds), n_train)
        kf = KFold(n_splits=folds, shuffle=True, random_state=int(seed))

        for weights in ["uniform", "distance"]:
            for k in k_values:
                fold_rhos: list[float] = []
                fold_rmses: list[float] = []
                space_fold_rhos: dict[str, list[float]] = {view_id: [] for view_id in split.view_ids}
                space_fold_rmses: dict[str, list[float]] = {view_id: [] for view_id in split.view_ids}

                for fold_train_idx, fold_val_idx in kf.split(np.arange(n_train)):
                    if k > len(fold_train_idx):
                        continue

                    y_fold_train = split.y_train[fold_train_idx]
                    y_fold_val = split.y_train[fold_val_idx]
                    fold_space_rhos: list[float] = []
                    fold_space_rmses: list[float] = []

                    for view_id in split.view_ids:
                        model = create_regressor(
                            kind=model_kind,
                            n_neighbors=int(k),
                            weights=str(weights),
                            n_jobs=n_jobs,
                        )
                        model.fit(split.x_train_by_view[view_id][fold_train_idx], y_fold_train)
                        y_pred = model.predict(split.x_train_by_view[view_id][fold_val_idx])
                        rho_view = safe_spearman(y_fold_val, y_pred)
                        rmse_view = rmse(y_fold_val, y_pred)
                        space_fold_rhos[view_id].append(float(rho_view))
                        space_fold_rmses[view_id].append(float(rmse_view))
                        fold_space_rhos.append(float(rho_view))
                        fold_space_rmses.append(float(rmse_view))

                    fold_rhos.append(self._mean(fold_space_rhos))
                    fold_rmses.append(self._mean(fold_space_rmses))

                if not fold_rhos:
                    continue

                row: dict = {
                    "k": int(k),
                    "weights": str(weights),
                    "mean_spearman": float(np.mean(fold_rhos)),
                    "std_spearman": float(np.std(fold_rhos, ddof=1)) if len(fold_rhos) > 1 else 0.0,
                    "mean_rmse": float(np.mean(fold_rmses)),
                    "std_rmse": float(np.std(fold_rmses, ddof=1)) if len(fold_rmses) > 1 else 0.0,
                }
                for view_id in split.view_ids:
                    rhos = space_fold_rhos[view_id]
                    rmses = space_fold_rmses[view_id]
                    row[f"mean_spearman__{view_id}"] = float(np.mean(rhos))
                    row[f"std_spearman__{view_id}"] = float(np.std(rhos, ddof=1)) if len(rhos) > 1 else 0.0
                    row[f"mean_rmse__{view_id}"] = float(np.mean(rmses))
                    row[f"std_rmse__{view_id}"] = float(np.std(rmses, ddof=1)) if len(rmses) > 1 else 0.0
                rows.append(row)

        return rows

    def select_params(
        self,
        prepared: PredictionPreparedData,
        split: PredictionSplitData,
        config: dict,
    ) -> tuple[list[dict], dict, int]:
        """Return ranked CV rows, the selected row, and resolved worker count.

        Rows are ranked by descending mean Spearman correlation and then
        ascending mean RMSE.
        """
        prediction_cfg = config["prediction"]
        model_cfg = config["model"]
        runtime_cfg = config["runtime"]

        n_train = len(split.y_train)
        cv_folds = min(int(prediction_cfg["cv_folds"]), n_train)
        k_values = self._k_values(
            n_train=n_train,
            cv_folds=cv_folds,
            prediction_cfg=prediction_cfg,
        )
        n_jobs = resolve_n_jobs(int(runtime_cfg["n_jobs"]))

        if prepared.embedding_mode == "multi_space":
            cv_rows = self._multi_space_cv_rows(
                split=split,
                seed=int(runtime_cfg["seed"]),
                cv_folds=cv_folds,
                k_values=k_values,
                model_kind=str(model_cfg["kind"]),
                n_jobs=n_jobs,
            )
        else:
            view_id = split.view_ids[0]
            cv_results = cross_validate_knn(
                x=split.x_train_by_view[view_id],
                y=split.y_train,
                k_values=k_values,
                weights_options=["uniform", "distance"],
                n_splits=cv_folds,
                seed=int(runtime_cfg["seed"]),
                model_kind=str(model_cfg["kind"]),
                n_jobs=n_jobs,
            )
            if not cv_results:
                raise ValueError("Cross-validation produced no results.")

            cv_rows = []
            for row in cv_results:
                cv_rows.append(
                    {
                        "k": int(row.k),
                        "weights": str(row.weights),
                        "mean_spearman": float(row.mean_spearman),
                        "std_spearman": float(row.std_spearman),
                        "mean_rmse": float(row.mean_rmse),
                        "std_rmse": float(row.std_rmse),
                        f"mean_spearman__{view_id}": float(row.mean_spearman),
                        f"std_spearman__{view_id}": float(row.std_spearman),
                        f"mean_rmse__{view_id}": float(row.mean_rmse),
                        f"std_rmse__{view_id}": float(row.std_rmse),
                    }
                )

        if not cv_rows:
            raise ValueError("Cross-validation produced no results.")

        rows_sorted = sorted(cv_rows, key=lambda row: (-row["mean_spearman"], row["mean_rmse"]))
        return rows_sorted, rows_sorted[0], n_jobs

    def _predict_vocab_rows(
        self,
        view_ids: list[str],
        reference_words: list[str],
        reference_scores: np.ndarray,
        models_by_view: dict[str, RegressorBackend],
        vocab_words: list[str],
        x_vocab_by_view: dict[str, np.ndarray],
        topn_neighbors: int,
        k_for_prediction: int,
        prediction_weights: str,
    ) -> tuple[list[dict], dict[str, float | int]]:
        """Build target rows with predictions and self-excluded evidence.

        One extra neighbor is fetched so a target that is also a reference word
        can exclude itself without reducing the requested evidence or prediction
        neighborhood. Predictions affected by self-exclusion are recomputed from
        the filtered neighbor scores and distances.
        """
        preds_by_view: dict[str, np.ndarray] = {}
        neighbors_by_view: dict[str, np.ndarray] = {}
        distances_by_view: dict[str, np.ndarray] = {}
        reference_word_set = set(reference_words)
        fetch_k = min(len(reference_words), max(int(topn_neighbors), int(k_for_prediction)) + 1)

        for view_id, model in models_by_view.items():
            preds, topk_idx, topk_dist = model.predict_with_neighbors(
                x_target=x_vocab_by_view[view_id],
                topn=int(fetch_k),
            )
            preds_by_view[view_id] = preds
            neighbors_by_view[view_id] = topk_idx
            distances_by_view[view_id] = topk_dist

        primary_view = view_ids[0]

        rows: list[dict] = []
        n_self_filtered_rows = 0
        for i, word in enumerate(vocab_words):
            row_self_filtered = False
            row_pred_by_view: dict[str, float] = {}
            row_neighbor_ids: dict[str, np.ndarray] = {}
            row_neighbor_dist: dict[str, np.ndarray] = {}

            for view_id in view_ids:
                ids = neighbors_by_view[view_id][i]
                dist = distances_by_view[view_id][i]
                if word in reference_word_set:
                    keep_mask = np.fromiter(
                        (reference_words[int(j)] != word for j in ids),
                        dtype=bool,
                        count=len(ids),
                    )
                    if not bool(np.all(keep_mask)):
                        row_self_filtered = True
                else:
                    keep_mask = np.ones(len(ids), dtype=bool)

                filtered_ids = ids[keep_mask]
                filtered_dist = dist[keep_mask]
                if filtered_ids.size == 0:
                    raise ValueError(f"No neighbors left after self-filtering for vocab word '{word}'.")

                row_neighbor_ids[view_id] = filtered_ids
                row_neighbor_dist[view_id] = filtered_dist

                if bool(np.all(keep_mask)):
                    row_pred_by_view[view_id] = float(preds_by_view[view_id][i])
                else:
                    k_eff = min(int(k_for_prediction), int(filtered_ids.size))
                    score_slice = reference_scores[filtered_ids[:k_eff]]
                    dist_slice = filtered_dist[:k_eff]
                    row_pred_by_view[view_id] = self._weighted_knn_prediction(
                        scores=np.asarray(score_slice, dtype=np.float32),
                        distances=np.asarray(dist_slice, dtype=np.float32),
                        weights=str(prediction_weights),
                    )

            if row_self_filtered:
                n_self_filtered_rows += 1

            primary_ids = row_neighbor_ids[primary_view][: int(topn_neighbors)]
            primary_dists = row_neighbor_dist[primary_view][: int(topn_neighbors)]
            primary_words = [reference_words[int(j)] for j in primary_ids]
            primary_scores = [float(reference_scores[int(j)]) for j in primary_ids]
            primary_dist = [float(v) for v in primary_dists]
            row: dict = {
                "word": word,
                "is_gold_observed": bool(word in reference_word_set),
                "pred": float(np.mean([row_pred_by_view[view_id] for view_id in view_ids])),
                "neighbors": " | ".join(primary_words),
                "neighbor_gold_scores": " | ".join(f"{v:.4f}" for v in primary_scores),
                "neighbor_cosine_distances": " | ".join(f"{v:.6f}" for v in primary_dist),
            }

            for view_id in view_ids:
                ids = row_neighbor_ids[view_id][: int(topn_neighbors)]
                dist = row_neighbor_dist[view_id][: int(topn_neighbors)]
                n_words = [reference_words[int(j)] for j in ids]
                n_scores = [float(reference_scores[int(j)]) for j in ids]
                n_dist = [float(v) for v in dist]
                row[f"pred__{view_id}"] = float(row_pred_by_view[view_id])
                row[f"neighbors__{view_id}"] = " | ".join(n_words)
                row[f"neighbor_gold_scores__{view_id}"] = " | ".join(f"{v:.4f}" for v in n_scores)
                row[f"neighbor_cosine_distances__{view_id}"] = " | ".join(f"{v:.6f}" for v in n_dist)
            rows.append(row)

        n_vocab_rows = int(len(vocab_words))
        return rows, {
            "n_vocab_rows": n_vocab_rows,
            "n_self_filtered_rows": int(n_self_filtered_rows),
            "self_filtered_ratio": float(n_self_filtered_rows / max(1, n_vocab_rows)),
            "n_vocab_reference_words": int(len(reference_words)),
        }

    def train(
        self,
        prepared: PredictionPreparedData,
        split: PredictionSplitData,
        config: dict,
        vocab_features: tuple[list[str], dict[str, np.ndarray], list[str]] | None = None,
    ) -> PredictionTrainingResult:
        """Select settings, evaluate the outer test split, and predict targets.

        Outer-test models are fitted on ``split.x_train_by_view``. Target models
        are created only when ``vocab_features`` is present and are fitted on
        ``prepared.x_all_by_view`` and ``prepared.y_kept``—all gold items with
        embedding coverage. This post-evaluation refit does not alter reported
        test metrics or the selected settings.
        """
        model_cfg = config["model"]
        prediction_cfg = config["prediction"]

        cv_rows, best, n_jobs = self.select_params(prepared=prepared, split=split, config=config)

        models_by_view: dict[str, RegressorBackend] = {}
        preds_by_view: dict[str, np.ndarray] = {}
        test_metrics_by_view: dict[str, dict[str, float]] = {}

        for view_id in split.view_ids:
            model = create_regressor(
                kind=str(model_cfg["kind"]),
                n_neighbors=int(best["k"]),
                weights=str(best["weights"]),
                n_jobs=n_jobs,
            )
            model.fit(split.x_train_by_view[view_id], split.y_train)
            y_pred = model.predict(split.x_test_by_view[view_id])
            models_by_view[view_id] = model
            preds_by_view[view_id] = y_pred
            test_metrics_by_view[view_id] = {
                "spearman": float(safe_spearman(split.y_test, y_pred)),
                "rmse": float(rmse(split.y_test, y_pred)),
            }

        merged_test_pred = np.mean(
            np.column_stack([preds_by_view[view_id] for view_id in split.view_ids]),
            axis=1,
        )

        result = PredictionTrainingResult(
            cv_rows=cv_rows,
            best_params=best,
            n_jobs=int(n_jobs),
            models_by_view=models_by_view,
            test_predictions_by_view=preds_by_view,
            test_metrics_by_view=test_metrics_by_view,
            merged_test_predictions=merged_test_pred,
            test_spearman=float(safe_spearman(split.y_test, merged_test_pred)),
            test_rmse=float(rmse(split.y_test, merged_test_pred)),
        )

        if vocab_features is not None:
            vocab_words, x_vocab_by_view, dropped_vocab_words = vocab_features
            vocab_models_by_view: dict[str, RegressorBackend] = {}
            for view_id in split.view_ids:
                vocab_model = create_regressor(
                    kind=str(model_cfg["kind"]),
                    n_neighbors=int(best["k"]),
                    weights=str(best["weights"]),
                    n_jobs=n_jobs,
                )
                vocab_model.fit(prepared.x_all_by_view[view_id], prepared.y_kept)
                vocab_models_by_view[view_id] = vocab_model

            vocab_rows, vocab_filter_stats = self._predict_vocab_rows(
                view_ids=split.view_ids,
                reference_words=prepared.words_kept,
                reference_scores=prepared.y_kept,
                models_by_view=vocab_models_by_view,
                vocab_words=vocab_words,
                x_vocab_by_view=x_vocab_by_view,
                topn_neighbors=int(prediction_cfg["topn_neighbors"]),
                k_for_prediction=int(best["k"]),
                prediction_weights=str(best["weights"]),
            )
            result.vocab_rows = vocab_rows
            result.dropped_vocab_words = dropped_vocab_words
            result.vocab_self_filter_stats = vocab_filter_stats
            result.vocab_fit_scope = "all_covered_gold_post_eval"
            result.n_vocab_reference_words = int(len(prepared.words_kept))

        return result
