# Synthetic smoke example

Every value in this directory is invented for software testing. The example is
small enough to run in seconds and does not represent a scientifically useful
lexical resource.

From the repository root, run:

```bash
concreteness-knn-core prediction run --config examples/smoke/config.json
concreteness-knn-core prediction holdout --config examples/smoke/config.json
```
