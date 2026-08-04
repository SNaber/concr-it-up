# Glossary

- **Covered gold item**: A seed-lexicon item for which the active embedding
  configuration provides a vector.
- **CV**: Cross-validation within the outer training partition, used to select
  the number of neighbors and weighting mode.
- **Held-out test partition**: The portion of the seed lexicon reserved for
  evaluation after parameter selection.
- **External holdout**: A separate ratings file used to score an existing
  prediction CSV.
- **OOV**: A word for which the active embedding configuration cannot provide a
  vector.
- **Self-filtering**: Removal of a target word from its own displayed neighbor
  evidence when that word also occurs in the seed lexicon.
- **Core report level**: Evaluation summary, CV results, and held-out test
  predictions.
- **Full report level**: Core outputs plus target predictions and OOV
  diagnostics.
