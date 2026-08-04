# Configuration reference

A configuration is a single JSON object merged over defaults from
[`default_config()`](../src/concreteness_knn_core/config.py).

## Sections

- `dataset`: gold ratings, column names, normalization, and optional POS filter;
- `embeddings`: one or more embedding spaces and the active combination mode;
- `model`: regression backend;
- `runtime`: random seed, worker count, and output directory;
- `prediction`: split, CV, neighbor, target-vocabulary, and holdout settings;
- `reports`: core or full artifact level.

## Validation tasks

- `prediction_run`
- `prediction_holdout`

## Generated key table

The canonical key/default table is
[generated_config_reference.md](generated_config_reference.md). It is generated
from the implementation; regenerate it after configuration changes.
