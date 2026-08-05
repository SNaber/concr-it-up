from __future__ import annotations

import io
import json
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from apps.core_gui.app import create_app
from apps.core_gui.hosted_config import owner_storage_key
from apps.core_gui.session_storage import ACTIVITY_FILENAME, cleanup_expired_session_storage
from apps.core_gui.worker import JobWorker, WorkerSettings


HTTPS_ROOT = "https://localhost"


def _csrf(client) -> str:
    response = client.get("/", base_url=HTTPS_ROOT)
    assert response.status_code == 200
    match = re.search(r'<meta name="csrf-token" content="([^"]+)"', response.get_data(as_text=True))
    assert match is not None
    return match.group(1)


def _post(client, url: str, csrf: str, **kwargs):
    headers = dict(kwargs.pop("headers", {}))
    headers["X-CSRF-Token"] = csrf
    return client.post(url, base_url=HTTPS_ROOT, headers=headers, **kwargs)


@pytest.fixture()
def hosted_app(tmp_path: Path, monkeypatch):
    embedding = tmp_path / "embeddings" / "mini.vec"
    embedding.parent.mkdir(parents=True)
    embedding.write_text("1 2\nword 1.0 0.0\n", encoding="utf-8")
    paper_configs = tmp_path / "configs" / "paper_runs"
    paper_configs.mkdir(parents=True)
    (paper_configs / "english.json").write_text("{}\n", encoding="utf-8")

    allowlist = {
        "embeddings": [
            {
                "id": "mini",
                "kind": "vec",
                "path": str(embedding),
                "label": "Synthetic mini vectors",
            }
        ]
    }
    monkeypatch.setenv("CONCRITUP_HOSTED_MODE", "1")
    monkeypatch.setenv("CONCRITUP_ACCESS_MODE", "anonymous")
    monkeypatch.setenv("CONCRITUP_SECRET_KEY", "s" * 64)
    monkeypatch.setenv("CONCRITUP_EMBEDDING_ALLOWLIST", json.dumps(allowlist))
    monkeypatch.setenv("CONCRITUP_JOB_DB", str(tmp_path / "state" / "jobs.sqlite3"))
    monkeypatch.setenv("CONCRITUP_JOB_ROOT", str(tmp_path / "jobs"))
    monkeypatch.setenv(
        "CONCRITUP_SESSION_ROOT", str(tmp_path / "data" / "hosted_sessions")
    )
    monkeypatch.setenv("CONCRITUP_PAPER_CONFIG_ROOT", str(paper_configs))

    return create_app(repo_root=tmp_path, testing=True, start_embedded_worker=False)


def _upload_inputs(client, csrf: str) -> tuple[str, str]:
    gold = _post(
        client,
        "/api/fs/upload",
        csrf,
        data={
            "field_target": "gold",
            "file": (io.BytesIO(b"Word,Conc.M\na,1\nb,2\nc,3\nd,4\ne,5\nf,2.5\n"), "gold.csv"),
        },
        content_type="multipart/form-data",
    )
    assert gold.status_code == 201
    target = _post(
        client,
        "/api/fs/upload",
        csrf,
        data={
            "field_target": "target",
            "file": (io.BytesIO(b"a\nb\nc\n"), "target.txt"),
        },
        content_type="multipart/form-data",
    )
    assert target.status_code == 201
    return gold.get_json()["path"], target.get_json()["path"]


def _hosted_config(client, gold: str, target: str) -> dict:
    response = client.get("/api/config/default", base_url=HTTPS_ROOT)
    assert response.status_code == 200
    config = response.get_json()["raw_config"]
    config["dataset"]["gold"] = gold
    config["dataset"]["word_column"] = "Word"
    config["dataset"]["score_column"] = "Conc.M"
    config["dataset"]["pos_filter"]["enabled"] = False
    config["prediction"].update(
        {
            "target": target,
            "test_size": 0.2,
            "cv_folds": 2,
            "k_min": 1,
            "k_max": 2,
            "k_step": 1,
            "topn_neighbors": 2,
        }
    )
    return config


def test_anonymous_cookie_csrf_virtual_paths_and_allowlist(hosted_app):
    client = hosted_app.test_client()
    index = client.get("/", base_url=HTTPS_ROOT)
    cookie = index.headers.get("Set-Cookie", "")
    assert "Secure" in cookie
    assert "HttpOnly" in cookie
    assert "SameSite=Lax" in cookie
    rendered = index.get_data(as_text=True)
    assert 'id="retentionNotice"' in rendered
    assert "48 hours of inactivity" in rendered
    assert "48 hours after completion or failure" in rendered
    assert "Download anything you need to keep" in rendered
    assert 'id="uploadPrivacyNotice"' in rendered
    assert "Do not upload personal" in rendered
    assert 'href="/impressum"' in rendered
    assert 'href="/datenschutz"' in rendered

    rejected = client.post("/api/config/save", base_url=HTTPS_ROOT, json={})
    assert rejected.status_code == 403

    csrf = _csrf(client)
    embeddings = client.get("/api/fs/embeddings", base_url=HTTPS_ROOT).get_json()["candidates"]
    assert embeddings == [
        {"label": "Synthetic mini vectors", "name": "mini.vec", "path": "embedding:mini"}
    ]
    assert str(hosted_app.config["GUI_REPO_ROOT"]) not in json.dumps(embeddings)

    listing = client.get("/api/fs/list", base_url=HTTPS_ROOT, query_string={"path": "configs"})
    assert listing.status_code == 200
    assert {entry["path"] for entry in listing.get_json()["entries"]} == {
        "configs/paper_runs",
        "configs/user",
    }
    outside = client.get("/api/fs/list", base_url=HTTPS_ROOT, query_string={"path": "/etc"})
    assert outside.status_code == 400

    saved = _post(
        client,
        "/api/config/save",
        csrf,
        json={"path": "configs/user/mine.json", "config": {"dataset": {"gold": "upload:x"}}},
    )
    assert saved.status_code == 200
    assert saved.get_json()["path"] == "session-config:mine.json"


def test_hosted_request_refreshes_owner_storage_activity(hosted_app):
    client = hosted_app.test_client()
    csrf = _csrf(client)
    with client.session_transaction() as browser_session:
        owner = browser_session["anonymous_owner"]

    hosted = hosted_app.config["HOSTED_SETTINGS"]
    owner_root = hosted.session_root / owner_storage_key(owner)
    assert not owner_root.exists()

    uploaded = _post(
        client,
        "/api/fs/upload",
        csrf,
        data={
            "field_target": "target",
            "file": (io.BytesIO(b"a\n"), "target.txt"),
        },
        content_type="multipart/form-data",
    )
    assert uploaded.status_code == 201
    marker = owner_root / ACTIVITY_FILENAME
    old_timestamp = 1_000_000_000
    os.utime(marker, (old_timestamp, old_timestamp))

    assert client.get("/api/config/default", base_url=HTTPS_ROOT).status_code == 200
    assert marker.stat().st_mtime > old_timestamp


def test_read_only_anonymous_visits_do_not_create_owner_trees(hosted_app) -> None:
    hosted = hosted_app.config["HOSTED_SETTINGS"]
    for _ in range(20):
        assert hosted_app.test_client().get("/", base_url=HTTPS_ROOT).status_code == 200

    owner_trees = (
        [
            child
            for child in hosted.session_root.iterdir()
            if child.is_dir() and re.fullmatch(r"[0-9a-f]{64}", child.name)
        ]
        if hosted.session_root.exists()
        else []
    )
    assert owner_trees == []


def test_public_legal_pages_are_cookie_free_and_do_not_create_owner_storage(hosted_app) -> None:
    hosted = hosted_app.config["HOSTED_SETTINGS"]
    client = hosted_app.test_client()

    for path in ("/impressum", "/datenschutz", "/static/styles.css"):
        response = client.get(path, base_url=HTTPS_ROOT)
        assert response.status_code == 200
        assert "concritup_session" not in response.headers.get("Set-Cookie", "")

    privacy_html = client.get("/datenschutz", base_url=HTTPS_ROOT).get_data(as_text=True)
    assert "48 Stunden ohne Aktivität" in privacy_html
    assert "48 Stunden nach Abschluss" in privacy_html
    assert str(hosted.session_root) not in privacy_html
    assert str(hosted.job_root) not in privacy_html

    owner_trees = (
        [
            child
            for child in hosted.session_root.iterdir()
            if child.is_dir() and re.fullmatch(r"[0-9a-f]{64}", child.name)
        ]
        if hosted.session_root.exists()
        else []
    )
    assert owner_trees == []


def test_legal_pages_bypass_authenticated_staging_identity(tmp_path: Path, monkeypatch) -> None:
    embedding = tmp_path / "embeddings" / "mini.vec"
    embedding.parent.mkdir(parents=True)
    embedding.write_text("1 2\nword 1.0 0.0\n", encoding="utf-8")
    paper_configs = tmp_path / "configs" / "paper_runs"
    paper_configs.mkdir(parents=True)

    monkeypatch.setenv("CONCRITUP_HOSTED_MODE", "1")
    monkeypatch.setenv("CONCRITUP_ACCESS_MODE", "authenticated")
    monkeypatch.setenv("CONCRITUP_SECRET_KEY", "s" * 64)
    monkeypatch.setenv(
        "CONCRITUP_EMBEDDING_ALLOWLIST",
        json.dumps(
            {
                "embeddings": [
                    {
                        "id": "mini",
                        "kind": "vec",
                        "path": str(embedding),
                        "label": "Synthetic mini vectors",
                    }
                ]
            }
        ),
    )
    monkeypatch.setenv("CONCRITUP_JOB_DB", str(tmp_path / "state" / "jobs.sqlite3"))
    monkeypatch.setenv("CONCRITUP_JOB_ROOT", str(tmp_path / "jobs"))
    monkeypatch.setenv("CONCRITUP_SESSION_ROOT", str(tmp_path / "data" / "hosted_sessions"))
    monkeypatch.setenv("CONCRITUP_PAPER_CONFIG_ROOT", str(paper_configs))

    app = create_app(repo_root=tmp_path, testing=True, start_embedded_worker=False)
    client = app.test_client()

    assert client.get("/", base_url=HTTPS_ROOT).status_code == 401
    assert client.get("/impressum", base_url=HTTPS_ROOT).status_code == 200
    assert client.get("/datenschutz", base_url=HTTPS_ROOT).status_code == 200
    assert client.get("/static/styles.css", base_url=HTTPS_ROOT).status_code == 200


def test_early_csrf_rejection_releases_session_cleanup_lock(hosted_app) -> None:
    client = hosted_app.test_client()
    csrf = _csrf(client)
    _upload_inputs(client, csrf)
    with client.session_transaction() as browser_session:
        owner = browser_session["anonymous_owner"]
    hosted = hosted_app.config["HOSTED_SETTINGS"]
    key = owner_storage_key(owner)
    owner_root = hosted.session_root / key

    rejected = client.post("/api/config/save", base_url=HTTPS_ROOT, json={})
    assert rejected.status_code == 403
    reference = datetime.now(timezone.utc)
    stale = (reference - timedelta(days=7)).timestamp()
    os.utime(owner_root / ACTIVITY_FILENAME, (stale, stale))

    assert cleanup_expired_session_storage(
        hosted.session_root,
        retention_hours=48,
        now=reference,
    ) == [key]


def test_anonymous_jobs_uploads_and_results_are_session_scoped(hosted_app):
    first = hosted_app.test_client()
    second = hosted_app.test_client()
    first_csrf = _csrf(first)
    second_csrf = _csrf(second)
    gold, target = _upload_inputs(first, first_csrf)
    config = _hosted_config(first, gold, target)

    submitted = _post(
        first,
        "/api/jobs/prediction-run",
        first_csrf,
        json={"config": config},
    )
    assert submitted.status_code == 202
    job_id = submitted.get_json()["job_id"]

    own = first.get(f"/api/jobs/{job_id}", base_url=HTTPS_ROOT)
    assert own.status_code == 200
    assert own.get_json()["status"] == "queued"
    assert "owner" not in own.get_json()
    assert "config_path" not in own.get_json()

    cross_session = second.get(f"/api/jobs/{job_id}", base_url=HTTPS_ROOT)
    assert cross_session.status_code == 404
    cross_download = second.get(
        "/api/results/download",
        base_url=HTTPS_ROOT,
        query_string={"job_id": job_id, "artifact_key": "summary"},
    )
    assert cross_download.status_code == 404

    duplicate = _post(
        first,
        "/api/jobs/prediction-run",
        first_csrf,
        json={"config": config},
    )
    assert duplicate.status_code == 409

    saved = _post(
        first,
        "/api/config/save",
        first_csrf,
        json={"path": "session-config:mine.json", "config": config},
    )
    assert saved.status_code == 200

    stolen_config = _hosted_config(second, gold, target)
    stolen = _post(
        second,
        "/api/jobs/prediction-run",
        second_csrf,
        json={"config": stolen_config},
    )
    assert stolen.status_code == 400

    saved_cross = _post(
        second,
        "/api/config/load",
        second_csrf,
        json={"path": "session-config:mine.json"},
    )
    assert saved_cross.status_code == 400


def test_hosted_file_upload_limit_is_enforced(hosted_app):
    client = hosted_app.test_client()
    csrf = _csrf(client)
    oversized = b"x" * ((10 * 1024 * 1024) + 1)
    response = _post(
        client,
        "/api/fs/upload",
        csrf,
        data={
            "field_target": "target",
            "file": (io.BytesIO(oversized), "too-large.txt"),
        },
        content_type="multipart/form-data",
    )
    assert response.status_code == 413
    with client.session_transaction() as browser_session:
        owner = browser_session["anonymous_owner"]
    hosted = hosted_app.config["HOSTED_SETTINGS"]
    assert not (hosted.session_root / owner_storage_key(owner)).exists()


def test_hosted_uploads_and_saved_configs_stop_at_disk_admission_limit(
    hosted_app, monkeypatch
):
    client = hosted_app.test_client()
    csrf = _csrf(client)
    store = hosted_app.extensions["concritup_job_store"]
    monkeypatch.setattr(store, "admission_available", lambda: False)

    upload = _post(
        client,
        "/api/fs/upload",
        csrf,
        data={
            "field_target": "target",
            "file": (io.BytesIO(b"word\n"), "target.txt"),
        },
        content_type="multipart/form-data",
    )
    assert upload.status_code == 409
    assert "80%" in upload.get_json()["error"]

    saved = _post(
        client,
        "/api/config/save",
        csrf,
        json={"path": "session-config:blocked.json", "config": {}},
    )
    assert saved.status_code == 409
    assert "80%" in saved.get_json()["error"]
    with client.session_transaction() as browser_session:
        owner = browser_session["anonymous_owner"]
    hosted = hosted_app.config["HOSTED_SETTINGS"]
    assert not (hosted.session_root / owner_storage_key(owner)).exists()


def test_completed_artifacts_and_manifest_remain_owner_scoped_and_path_safe(hosted_app):
    owner_client = hosted_app.test_client()
    other_client = hosted_app.test_client()
    csrf = _csrf(owner_client)
    _csrf(other_client)
    gold, target = _upload_inputs(owner_client, csrf)
    config = _hosted_config(owner_client, gold, target)
    submitted = _post(
        owner_client,
        "/api/jobs/prediction-run",
        csrf,
        json={"config": config},
    )
    assert submitted.status_code == 202
    job_id = submitted.get_json()["job_id"]

    hosted = hosted_app.config["HOSTED_SETTINGS"]
    store = hosted_app.extensions["concritup_job_store"]
    settings = WorkerSettings(
        repo_root=hosted.repo_root,
        db_path=hosted.db_path,
        job_root=hosted.job_root,
        entrypoint="concreteness-knn-core",
        worker_id="hosted-test-worker",
    )

    def fake_core(job, _command, _cwd, _environment):
        (job.paths.output_dir / "summary.json").write_text(
            json.dumps({"best_k": 1, "best_weights": "distance"}) + "\n",
            encoding="utf-8",
        )
        (job.paths.output_dir / "cv_results.csv").write_text(
            "k,weights,mean_spearman,mean_rmse\n1,distance,0.5,0.5\n",
            encoding="utf-8",
        )
        (job.paths.output_dir / "test_predictions.csv").write_text(
            "word,gold,pred,error,abs_error\na,1,1,0,0\n",
            encoding="utf-8",
        )
        (job.paths.output_dir / "vocab_predictions.csv").write_text(
            "word,pred\na,1\n",
            encoding="utf-8",
        )
        return 0

    worker = JobWorker(store, settings, executor=fake_core)
    assert worker.run_once() is True

    completed = owner_client.get(f"/api/jobs/{job_id}", base_url=HTTPS_ROOT)
    assert completed.status_code == 200
    body = completed.get_json()
    assert body["status"] == "done"
    assert all(value.startswith(f"job-artifact:{job_id}:") for value in body["result"]["artifacts"].values())
    assert str(hosted.repo_root) not in json.dumps(body)
    assert str(hosted.job_root) not in json.dumps(body)

    preview = owner_client.get(
        "/api/results/preview",
        base_url=HTTPS_ROOT,
        query_string={"job_id": job_id, "artifact_key": "vocab_predictions"},
    )
    assert preview.status_code == 200
    assert preview.get_json()["artifact_path"] == f"job-artifact:{job_id}:vocab_predictions"

    manifest_response = owner_client.get(
        "/api/results/download",
        base_url=HTTPS_ROOT,
        query_string={"job_id": job_id, "artifact_key": "run_manifest"},
    )
    assert manifest_response.status_code == 200
    rendered_manifest = manifest_response.get_data(as_text=True)
    assert str(hosted.repo_root) not in rendered_manifest
    assert str(hosted.job_root) not in rendered_manifest
    assert "anonymous_owner" not in rendered_manifest

    denied = other_client.get(
        "/api/results/download",
        base_url=HTTPS_ROOT,
        query_string={"job_id": job_id, "artifact_key": "run_manifest"},
    )
    assert denied.status_code == 404


def test_authenticated_staging_requires_proxy_identity_and_csrf(tmp_path: Path, monkeypatch):
    embedding = tmp_path / "mini.vec"
    embedding.write_text("1 2\nword 1 0\n", encoding="utf-8")
    paper_root = tmp_path / "paper"
    paper_root.mkdir()
    monkeypatch.setenv("CONCRITUP_HOSTED_MODE", "1")
    monkeypatch.setenv("CONCRITUP_ACCESS_MODE", "authenticated")
    monkeypatch.setenv("CONCRITUP_SECRET_KEY", "a" * 64)
    monkeypatch.setenv(
        "CONCRITUP_EMBEDDING_ALLOWLIST",
        json.dumps({"mini": {"kind": "vec", "path": str(embedding)}}),
    )
    monkeypatch.setenv("CONCRITUP_JOB_DB", str(tmp_path / "jobs.sqlite3"))
    monkeypatch.setenv("CONCRITUP_JOB_ROOT", str(tmp_path / "jobs"))
    monkeypatch.setenv(
        "CONCRITUP_SESSION_ROOT", str(tmp_path / "data" / "hosted_sessions")
    )
    monkeypatch.setenv("CONCRITUP_PAPER_CONFIG_ROOT", str(paper_root))
    app = create_app(repo_root=tmp_path, testing=True, start_embedded_worker=False)
    client = app.test_client()

    assert client.get("/healthz", base_url=HTTPS_ROOT).status_code == 200
    assert client.get("/", base_url=HTTPS_ROOT).status_code == 401
    index = client.get(
        "/",
        base_url=HTTPS_ROOT,
        headers={"X-Remote-User": "staging-reviewer"},
    )
    assert index.status_code == 200
    match = re.search(r'<meta name="csrf-token" content="([^"]+)"', index.get_data(as_text=True))
    assert match is not None
    csrf = match.group(1)

    missing_csrf = client.post(
        "/api/config/save",
        base_url=HTTPS_ROOT,
        headers={"X-Remote-User": "staging-reviewer"},
        json={},
    )
    assert missing_csrf.status_code == 403
    accepted_identity = client.get(
        "/api/config/default",
        base_url=HTTPS_ROOT,
        headers={"X-Remote-User": "staging-reviewer", "X-CSRF-Token": csrf},
    )
    assert accepted_identity.status_code == 200
