"""Filesystem writers for prediction and holdout artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from concreteness_knn_core.config import SCHEMA_VERSION

from .prediction_types import (
    PredictionHoldoutResult,
    PredictionPreparedData,
    PredictionSplitData,
    PredictionTrainingResult,
)


class PredictionReportWriter:
    """Write prediction-run and holdout-eval outputs to disk."""

    @staticmethod
    def _write_json(path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @staticmethod
    def _write_word_list(path: Path, words: list[str]) -> None:
        pd.DataFrame({"word": list(words)}).to_csv(path, index=False)

    def write_prediction_outputs(
        self,
        config: dict,
        config_path: str,
        prepared: PredictionPreparedData,
        split: PredictionSplitData,
        trained: PredictionTrainingResult,
    ) -> dict[str, Path]:
        """Persist prediction outputs for the requested report level."""
        runtime_cfg = config["runtime"]
        model_cfg = config["model"]
        reports_cfg = config["reports"]

        out_dir = Path(runtime_cfg["output_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        outputs: dict[str, Path] = {}

        cv_csv = out_dir / "cv_results.csv"
        pd.DataFrame(trained.cv_rows).to_csv(cv_csv, index=False)
        outputs["cv_results"] = cv_csv

        test_frame = pd.DataFrame(
            {
                "word": split.w_test,
                "gold": split.y_test,
                "pred": trained.merged_test_predictions,
                "error": trained.merged_test_predictions - split.y_test,
                "abs_error": np.abs(trained.merged_test_predictions - split.y_test),
            }
        )
        for view_id in split.view_ids:
            view_pred = trained.test_predictions_by_view[view_id]
            test_frame[f"pred__{view_id}"] = view_pred
            test_frame[f"error__{view_id}"] = view_pred - split.y_test
            test_frame[f"abs_error__{view_id}"] = np.abs(view_pred - split.y_test)

        test_csv = out_dir / "test_predictions.csv"
        test_frame.sort_values("abs_error", ascending=False).to_csv(test_csv, index=False)
        outputs["test_predictions"] = test_csv

        summary_path = out_dir / "summary.json"
        best = trained.best_params
        best_metrics_by_view = {
            view_id: {
                "mean_spearman": float(best[f"mean_spearman__{view_id}"]),
                "mean_rmse": float(best[f"mean_rmse__{view_id}"]),
            }
            for view_id in split.view_ids
        }

        summary_payload = {
            "schema_version": SCHEMA_VERSION,
            "config_path": str(config_path),
            "embedding_mode": prepared.embedding_mode,
            "embedding_space_ids": prepared.configured_space_ids,
            "prediction_space_ids": split.view_ids,
            "model_kind": model_cfg["kind"],
            "seed": int(runtime_cfg["seed"]),
            "n_jobs": int(trained.n_jobs),
            "n_gold_words": int(len(prepared.words_all)),
            "n_gold_with_embeddings": int(len(prepared.words_kept)),
            "gold_coverage": float(prepared.coverage),
            "n_train": int(len(split.y_train)),
            "n_test": int(len(split.y_test)),
            "best_k": int(best["k"]),
            "best_weights": str(best["weights"]),
            "cv_best_mean_spearman": float(best["mean_spearman"]),
            "cv_best_std_spearman": float(best["std_spearman"]),
            "cv_best_mean_rmse": float(best["mean_rmse"]),
            "cv_best_std_rmse": float(best["std_rmse"]),
            "cv_best_metrics_by_space": best_metrics_by_view,
            "test_spearman": float(trained.test_spearman),
            "test_rmse": float(trained.test_rmse),
            "test_metrics_by_space": trained.test_metrics_by_view,
            "report_level": reports_cfg["level"],
        }
        if trained.vocab_self_filter_stats is not None:
            summary_payload.update(trained.vocab_self_filter_stats)
        if trained.vocab_fit_scope is not None:
            summary_payload["vocab_fit_scope"] = str(trained.vocab_fit_scope)
        if trained.n_vocab_reference_words is not None:
            summary_payload["n_vocab_reference_words"] = int(trained.n_vocab_reference_words)

        self._write_json(summary_path, summary_payload)
        outputs["summary"] = summary_path

        if reports_cfg["level"] == "full":
            oov_gold_csv = out_dir / "oov_gold.csv"
            self._write_word_list(oov_gold_csv, prepared.dropped_gold_words)
            outputs["oov_gold"] = oov_gold_csv

            if trained.vocab_rows is not None:
                vocab_csv = out_dir / "vocab_predictions.csv"
                pd.DataFrame(trained.vocab_rows).to_csv(vocab_csv, index=False)
                outputs["vocab_predictions"] = vocab_csv

            oov_vocab_csv = out_dir / "oov_vocab.csv"
            self._write_word_list(oov_vocab_csv, trained.dropped_vocab_words or [])
            outputs["oov_vocab"] = oov_vocab_csv

        return outputs

    def write_holdout_outputs(
        self,
        config: dict,
        result: PredictionHoldoutResult,
    ) -> dict[str, Path]:
        """Persist holdout summary and optional unmatched diagnostics."""
        runtime_cfg = config["runtime"]
        prediction_cfg = config["prediction"]

        out_dir = Path(runtime_cfg["output_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)

        summary_path = out_dir / "holdout_summary.json"
        self._write_json(summary_path, result.summary_payload)
        outputs: dict[str, Path] = {"holdout_summary": summary_path}

        unmatched_out = prediction_cfg.get("unmatched_csv")
        if unmatched_out:
            holdout_only = sorted(set(result.holdout_df["_word"]) - set(result.merged_df["_word"]))
            pred_only = sorted(set(result.pred_df["_word"]) - set(result.merged_df["_word"]))
            rows = [{"source": "holdout_only", "word": word} for word in holdout_only]
            rows.extend({"source": "pred_only", "word": word} for word in pred_only)

            unmatched_path = Path(str(unmatched_out))
            unmatched_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(unmatched_path, index=False)
            outputs["holdout_unmatched"] = unmatched_path

        return outputs
