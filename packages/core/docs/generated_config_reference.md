# Generated Config Reference

Auto-generated from `default_config()` in `src/concreteness_knn_core/config.py`.

| Key | Type | Default | Valid/Notes |
| --- | --- | --- | --- |
| `dataset.gold` | `str` | `""` |  |
| `dataset.lowercase` | `bool` | `true` |  |
| `dataset.pos_filter.enabled` | `bool` | `false` |  |
| `dataset.pos_filter.pos_column` | `str` | `"Dom_Pos"` |  |
| `dataset.pos_filter.tags` | `list` | `["Noun"]` |  |
| `dataset.score_column` | `str` | `"Conc.M"` |  |
| `dataset.word_column` | `str` | `"Word"` |  |
| `embeddings.active_space` | `str` | `"default"` | required in single mode; member of spaces[*].id |
| `embeddings.mode` | `str` | `"single"` | single|joint|multi_space |
| `embeddings.spaces` | `list` | `[{"id": "default", "kind": "ft_bin", "label": "default_embedding", "path": ""}]` | non-empty list; unique ids; each kind in {ft_bin, vec} |
| `model.kind` | `str` | `"knn"` | knn |
| `prediction.cv_folds` | `int` | `5` | >=2 |
| `prediction.holdout` | `null` | `null` |  |
| `prediction.k_max` | `int` | `100` | >=1 and >= k_min |
| `prediction.k_min` | `int` | `5` | >=1 |
| `prediction.k_step` | `int` | `5` | >=1 |
| `prediction.prediction_column` | `str` | `"pred"` |  |
| `prediction.predictions_csv` | `null` | `null` |  |
| `prediction.target` | `null` | `null` |  |
| `prediction.test_size` | `float` | `0.2` | (0,1) |
| `prediction.topn_neighbors` | `int` | `10` | >=1 |
| `prediction.unmatched_csv` | `null` | `null` |  |
| `reports.level` | `str` | `"core"` | core|full |
| `runtime.n_jobs` | `int` | `-1` | nonzero |
| `runtime.output_dir` | `str` | `"data/generated_scores/und/prediction_run"` |  |
| `runtime.seed` | `int` | `13` |  |
