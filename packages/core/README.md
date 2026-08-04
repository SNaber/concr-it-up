# KNN-ConcrItUp core

`concreteness-knn-core` implements the configuration, data preparation, kNN
selection, evaluation, target prediction, and report-writing workflow used by
KNN-ConcrItUp. Although the paper evaluates concreteness, the regression target
may be another scalar lexical rating with an appropriate seed resource and
evaluation protocol.

## CLI

```bash
concreteness-knn-core prediction run --config <file>
concreteness-knn-core prediction holdout --config <file>
concreteness-knn-core config init --task prediction --out <file>
```

## Docs

See `docs/` for the workflow overview, CLI and configuration references,
prediction protocol, and output catalog.
