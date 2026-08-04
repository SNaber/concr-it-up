"""Prediction workflow orchestration and holdout scoring."""

from __future__ import annotations

import gc
from pathlib import Path

from concreteness_knn_core.config import SCHEMA_VERSION, validate_config

from .prediction_data_prep import PredictionDataPreparer
from .prediction_holdout_eval import PredictionHoldoutEvaluator
from .prediction_report_writer import PredictionReportWriter
from .prediction_trainer import PredictionTrainer


class PredictionPipeline:
    """Coordinate vectorization, model selection, evaluation, and reporting.

    A prediction run creates a fixed outer train/test split, selects kNN
    settings by cross-validation on the outer training portion, evaluates on
    the outer test portion, and then refits target-vocabulary models on every
    covered gold item. Holdout scoring is a separate artifact-matching step.
    """

    def __init__(
        self,
        data_prep: PredictionDataPreparer | None = None,
        trainer: PredictionTrainer | None = None,
        holdout_eval: PredictionHoldoutEvaluator | None = None,
        report_writer: PredictionReportWriter | None = None,
    ):
        """Initialize the pipeline with optional stage implementations."""
        self.data_prep = data_prep or PredictionDataPreparer()
        self.trainer = trainer or PredictionTrainer()
        self.holdout_eval = holdout_eval or PredictionHoldoutEvaluator()
        self.report_writer = report_writer or PredictionReportWriter()

    def run_prediction(self, config: dict, config_path: str) -> dict[str, Path]:
        """Run the configured prediction protocol and return artifact paths.

        Gold and optional target vectors are materialized before embedding
        models are released. Model selection and held-out evaluation use the
        outer split; target predictions, when requested, use a post-evaluation
        refit on all gold words with embeddings.
        """
        validate_config(config, task="prediction_run")

        prepared = self.data_prep.prepare_data(config)
        try:
            split = self.data_prep.split_data(prepared, config)

            vocab_features = None
            if config["reports"]["level"] == "full":
                vocab_features = self.data_prep.prepare_vocab_features(prepared, config)
        finally:
            # fastText models dominate resident memory. Releasing their owning
            # references after vectorization keeps CV and fitting memory-bounded.
            prepared.embeddings_by_space.clear()
            gc.collect()

        trained = self.trainer.train(
            prepared=prepared,
            split=split,
            config=config,
            vocab_features=vocab_features,
        )
        return self.report_writer.write_prediction_outputs(
            config=config,
            config_path=config_path,
            prepared=prepared,
            split=split,
            trained=trained,
        )

    def run_holdout(self, config: dict, config_path: str) -> dict[str, Path]:
        """Score predictions against holdout gold and write artifacts."""
        validate_config(config, task="prediction_holdout")
        result = self.holdout_eval.evaluate_holdout(
            config=config,
            config_path=config_path,
            schema_version=SCHEMA_VERSION,
        )
        return self.report_writer.write_holdout_outputs(config=config, result=result)
