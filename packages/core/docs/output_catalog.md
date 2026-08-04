# Output catalog

The canonical machine-generated artifact list is
[generated_output_manifest.md](generated_output_manifest.md). This catalog
describes how the artifacts are used.

## Prediction outputs

- `summary.json` records resolved run settings, coverage and split counts,
  selected hyperparameters, held-out metrics, and target-fit scope.
- `cv_results.csv` contains one row per evaluated neighbor-count and weighting
  combination, including fold-aggregated Spearman correlation and RMSE.
- `test_predictions.csv` contains predictions and errors for the held-out test
  partition used to report model quality.
- `vocab_predictions.csv`, written for a target vocabulary at full report level,
  contains predictions, gold-overlap flags, and nearest-neighbor evidence. Its
  model is refitted on all covered gold items after held-out evaluation.
- `oov_gold.csv` and `oov_vocab.csv` record gold and target words that could not
  be represented by the active embedding configuration.

## External-holdout outputs

- `holdout_summary.json` reports matching coverage, Spearman correlation, and
  RMSE for an existing prediction CSV and external ratings file.
- The optional unmatched CSV records words present on only one side of that
  comparison.
