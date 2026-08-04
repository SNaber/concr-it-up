# Prediction Pipeline

Implementation:

- [`prediction/prediction_pipeline.py`](../src/concreteness_knn_core/prediction/prediction_pipeline.py)
- [`prediction/prediction_data_prep.py`](../src/concreteness_knn_core/prediction/prediction_data_prep.py)
- [`prediction/prediction_trainer.py`](../src/concreteness_knn_core/prediction/prediction_trainer.py)
- [`prediction/prediction_holdout_eval.py`](../src/concreteness_knn_core/prediction/prediction_holdout_eval.py)
- [`prediction/prediction_report_writer.py`](../src/concreteness_knn_core/prediction/prediction_report_writer.py)
- [`prediction/prediction_types.py`](../src/concreteness_knn_core/prediction/prediction_types.py)

## `PredictionPipeline.run_prediction`

1. Validate config (`task="prediction_run"`).
2. Load and normalize the seed ratings, apply the optional POS filter, and
   retain covered gold items.
3. Load the configured embedding spaces and build the active feature views
   (`single`, `joint`, or `multi_space`). At full report level, also vectorize
   the target vocabulary and record OOV items.
4. Create the seeded outer train/test split. The test partition is not used for
   hyperparameter selection.
5. Within the outer training partition, run seeded cross-validation over the
   feasible `k` grid and both uniform and distance weighting. Select by highest
   mean Spearman correlation, using RMSE as the tie-breaker.
6. Fit the selected model on the outer training partition and evaluate it once
   on the held-out test partition.
7. For target-vocabulary prediction, refit the selected model on all covered
   gold items after evaluation. When a target word is also a gold item, exclude
   that word from its own prediction neighborhood and displayed evidence.
8. Write `summary.json`, `cv_results.csv`, `test_predictions.csv`, and the
   configured full-report artifacts.

## `PredictionPipeline.run_holdout`

1. Validate config (`task="prediction_holdout"`).
2. Load the external ratings file and prediction CSV.
3. Normalize and match words.
4. Compute Spearman and RMSE over matched rows.
5. Write `holdout_summary.json` and optional unmatched-word diagnostics.
