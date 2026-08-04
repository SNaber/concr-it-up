# Extending the core toolkit

## Add an embedding backend

1. Add an adapter in [`data/`](../src/concreteness_knn_core/data), preferably in
   a dedicated `data_embeddings_*.py` module.
2. Extend `load_embeddings(...)` dispatch.
3. Extend validation in [`config.py`](../src/concreteness_knn_core/config.py).

## Add a model backend

1. Add backend support in [`knn/`](../src/concreteness_knn_core/knn), preferably
   in a dedicated `knn_*.py` module.
2. Extend `create_regressor(...)` and config validation.
