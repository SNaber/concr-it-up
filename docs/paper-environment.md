# Paper environment

`requirements-paper.txt` pins the Python packages used for the paper runs. To
create that environment with Python 3.11:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install wheel==0.45.1
.venv/bin/python -m pip install --requirement requirements-paper.txt
.venv/bin/python -m pip install --no-deps --no-build-isolation --editable packages/core
```

After installation, run the synthetic example and the test suite. Reproducing
the eight paper runs also requires the separately licensed rating data, target
vocabularies, and embedding models described in `configs/paper_runs/`.

The requirements file fixes Python-package versions, but not the operating
system, BLAS implementation, CPU architecture, or external resource bytes.
Record the code revision and checksums of all inputs for each rerun; minor
platform-dependent floating-point differences may remain.
