# Core GUI

This package provides the KNN-ConcrItUp browser interface for the
`concreteness-knn-core` prediction and holdout workflows.

## Local use

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m apps.core_gui.app
```

The application is then available at <http://127.0.0.1:8000>. Local
development starts an embedded worker and executes each model run in a fresh
core CLI subprocess.

## Production use

Production deployments run the web process and model worker separately. Start
exactly one durable worker:

```bash
python -m apps.core_gui.worker
```

Serve the Flask application through a production WSGI server and reverse
proxy; do not expose the development server directly. The repository includes
example Gunicorn, systemd, Nginx, and environment templates under
`deploy/bwcloud/`. See `docs/deployment-bwcloud.md` for an overview.
