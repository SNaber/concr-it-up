# CLI reference

Executable:

```bash
concreteness-knn-core <group> <command> [options]
```

From `packages/core`, the module-mode equivalent is:

```bash
PYTHONPATH=src python -m concreteness_knn_core <group> <command> [options]
```

## Commands

### `prediction run`

Select hyperparameters, evaluate on a held-out split, refit on all covered gold
items, and optionally predict a target vocabulary.

```bash
concreteness-knn-core prediction run --config <file>
```

### `prediction holdout`

Compare an existing prediction CSV with an external holdout ratings file. This
command evaluates saved predictions; it does not train the model again.

```bash
concreteness-knn-core prediction holdout --config <file>
```

### `config init`

Write a default prediction configuration for editing.

```bash
concreteness-knn-core config init --task prediction --out <file>
```
