"""Prediction holdout matching, scoring, and overlap diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from concreteness_knn_core.knn import rmse, safe_spearman

from .prediction_types import PredictionHoldoutResult


class PredictionHoldoutEvaluator:
    """Evaluate prediction CSVs against holdout gold ratings."""

    @staticmethod
    def _raise_if_duplicate_words(words: pd.Series, source_name: str) -> None:
        """Reject normalized duplicate keys before the one-to-one merge."""
        dup_words = words.loc[words.duplicated(keep=False)]
        if dup_words.empty:
            return
        unique_dups = sorted(set(dup_words.tolist()))
        preview = ", ".join(unique_dups[:10])
        if len(unique_dups) > 10:
            preview += ", ..."
        raise ValueError(
            f"Duplicate words found in {source_name} after normalization: "
            f"{len(unique_dups)} duplicate key(s) [{preview}]"
        )

    def evaluate_holdout(
        self,
        config: dict,
        config_path: str,
        schema_version: str,
    ) -> PredictionHoldoutResult:
        """Match normalized word keys and compute holdout coverage and metrics.

        Duplicate normalized keys are rejected in either input so the merge is
        one-to-one. The returned payload retains the source and merged frames
        needed for unmatched-word reporting.
        """
        dataset_cfg = config["dataset"]
        prediction_cfg = config["prediction"]
        reports_cfg = config["reports"]

        holdout = pd.read_csv(str(prediction_cfg["holdout"]))
        pred = pd.read_csv(str(prediction_cfg["predictions_csv"]))

        word_column = str(dataset_cfg["word_column"])
        score_column = str(dataset_cfg["score_column"])
        prediction_column = str(prediction_cfg["prediction_column"])

        if word_column not in holdout.columns:
            raise ValueError(f"Word column '{word_column}' not found in {prediction_cfg['holdout']}.")
        if score_column not in holdout.columns:
            raise ValueError(f"Score column '{score_column}' not found in {prediction_cfg['holdout']}.")
        if "word" not in pred.columns:
            raise ValueError("Prediction CSV must contain a 'word' column.")
        if prediction_column not in pred.columns:
            raise ValueError(f"Prediction column '{prediction_column}' not found in {prediction_cfg['predictions_csv']}.")

        gold_word = holdout[word_column].astype(str).str.strip()
        pred_word = pred["word"].astype(str).str.strip()
        if bool(dataset_cfg.get("lowercase", True)):
            gold_word = gold_word.str.lower()
            pred_word = pred_word.str.lower()

        self._raise_if_duplicate_words(gold_word, "holdout")
        self._raise_if_duplicate_words(pred_word, "predictions_csv")

        holdout = holdout.assign(_word=gold_word)
        pred = pred.assign(_word=pred_word)
        merged = holdout.merge(pred, on="_word", how="inner", suffixes=("_gold", "_pred"))

        n_gold = int(len(holdout))
        n_pred = int(len(pred))
        n_matched = int(len(merged))
        overlap_ratio = float(n_matched / max(1, n_gold))
        if n_matched == 0:
            raise ValueError("No overlapping words between holdout and prediction files.")

        y_true = merged[score_column].astype(float).to_numpy(dtype=np.float32)
        y_hat = merged[prediction_column].astype(float).to_numpy(dtype=np.float32)
        rho = safe_spearman(y_true, y_hat)
        val_rmse = rmse(y_true, y_hat)

        print(
            f"[INFO] overlap: holdout={n_gold} pred={n_pred} matched={n_matched} "
            f"ratio={overlap_ratio:.4f}"
        )
        print(f"[HOLDOUT] spearman={rho:.4f} rmse={val_rmse:.4f} n={n_matched}")

        summary_payload: dict = {
            "schema_version": schema_version,
            "config_path": str(config_path),
            "holdout": str(prediction_cfg["holdout"]),
            "predictions_csv": str(prediction_cfg["predictions_csv"]),
            "prediction_column": prediction_column,
            "word_column": word_column,
            "score_column": score_column,
            "n_holdout_words": n_gold,
            "n_pred_words": n_pred,
            "n_matched": n_matched,
            "overlap_ratio": overlap_ratio,
            "spearman": float(rho),
            "rmse": float(val_rmse),
            "report_level": reports_cfg["level"],
        }

        return PredictionHoldoutResult(
            summary_payload=summary_payload,
            holdout_df=holdout,
            pred_df=pred,
            merged_df=merged,
        )
