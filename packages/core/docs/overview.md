# Overview

`concreteness_knn_core` extrapolates scalar lexical ratings from a rated seed
lexicon to a target vocabulary with k-nearest-neighbor regression over word
embeddings. It also scores existing predictions against an external ratings
file.

## Architecture

Core runtime modules:

- [`config.py`](../src/concreteness_knn_core/config.py)
- [`data/`](../src/concreteness_knn_core/data)
- [`knn/`](../src/concreteness_knn_core/knn)
- [`prediction/`](../src/concreteness_knn_core/prediction)
- [`cli.py`](../src/concreteness_knn_core/cli.py)

## Workflows

- `prediction run`: split the covered seed data, select hyperparameters by CV on
  the training partition, evaluate on the held-out partition, refit on all
  covered seed items for target prediction, and write artifacts.
- `prediction holdout`: compare an existing prediction CSV with an external
  ratings file without retraining.
