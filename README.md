# KNN-ConcrItUp

KNN-ConcrItUp is a browser-based workflow for extrapolating scalar lexical
ratings from a human-rated seed lexicon to a target vocabulary. It uses
k-nearest-neighbor regression over word embeddings, selects model settings by
cross-validation, evaluates them on a held-out split, and refits on all covered
gold items before predicting the target vocabulary. Predictions include their
nearest-neighbor evidence.

This repository contains the Flask interface and the
`concreteness-knn-core` backend described in *KNN-ConcrItUp! A
Language-Agnostic, Interactive, and Interpretable System for Extrapolating
Concreteness Norms*. Third-party rating datasets and embedding models are not
distributed with the software.

## Installation

Python 3.11 is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For development and tests:

```bash
python -m pip install -r requirements-dev.txt
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
```

`requirements-paper.txt` records the pinned package environment used for the
paper runs; see `docs/paper-environment.md`.

## Browser interface

From the repository root, run:

```bash
python -m apps.core_gui.app
```

Then open <http://127.0.0.1:8000>. The interface accepts a CSV/TSV gold file,
the word and score columns, an embedding model, and a target vocabulary with
one token per line. It provides evaluation summaries, cross-validation
results, vocabulary predictions, neighbor evidence, and OOV diagnostics.

The development server is intended for local use. A concise overview of the
included production templates is available in `docs/deployment-bwcloud.md`.

The imprint and privacy notice in `apps/core_gui/templates/` describe the
University of Stuttgart service. If you host your own instance, adapt these
pages to your operator, contact details, hostname, hosting, and data-handling
practices before making the service available.

## Command-line interface

```bash
concreteness-knn-core prediction run --config CONFIG.json
concreteness-knn-core prediction holdout --config CONFIG.json
concreteness-knn-core config init --task prediction --out CONFIG.json
```

Configuration and output documentation is under `packages/core/docs/`.

## Synthetic example

The self-contained example in `examples/smoke/` uses invented ratings and a
small word2vec-text embedding:

```bash
concreteness-knn-core prediction run --config examples/smoke/config.json
concreteness-knn-core prediction holdout --config examples/smoke/config.json
```

Its artifacts are written to the ignored directory
`data/generated_scores/smoke/`.

## Data and embeddings

Embedding files can be placed under `embeddings/`. Supported formats are
fastText `.bin` (`ft_bin`) and word2vec-style text `.vec` files. The eight
language-specific configurations in `configs/paper_runs/` document the runs
reported in the paper, but the licensed rating resources, target vocabularies,
and embedding models must be obtained separately. Reported settings and
results are summarized in `configs/paper_runs/manifest.csv`.

## Limitations

The generated values are model predictions, not new human ratings. Their
quality depends on the coverage and reliability of the seed resource and on
the representation and biases of the selected embedding model. Held-out
metrics, coverage summaries, OOV reports, and neighbor evidence should be
examined before using a resulting lexicon in research.

The paper configurations use different resources and rating scales, so their
metrics are not a controlled cross-language comparison. Extrapolation to other
scalar lexical variables requires separate evaluation of the target construct.

## Citation

If you use KNN-ConcrItUp, cite:

Sven Naber, Marina Aziz, Diego Frassinelli, and Sabine Schulte im Walde (2026).
*KNN-ConcrItUp! A Language-Agnostic, Interactive, and Interpretable System for
Extrapolating Concreteness Norms*.

## License

The software is released under the MIT License. Third-party datasets and
embedding models are not covered by this license.
