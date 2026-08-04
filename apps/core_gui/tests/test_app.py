from __future__ import annotations

import io
import json
import time
from pathlib import Path

import numpy as np
import pytest

from apps.core_gui.app import create_app


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, header: str, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")


def _write_vocab(path: Path, words: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(words) + "\n", encoding="utf-8")


def _write_existing_gui_artifacts(
    root: Path,
    *,
    include_prediction_core: bool = True,
    include_prediction_full: bool = True,
    include_vocab_predictions: bool = True,
    include_holdout_summary: bool = True,
    include_holdout_unmatched: bool = True,
) -> None:
    out_dir = root / "outputs" / "gui_run"
    out_dir.mkdir(parents=True, exist_ok=True)

    if include_prediction_core:
        _write_json(
            out_dir / "summary.json",
            {
                "schema_version": "7.0.0",
                "test_spearman": 0.42,
                "test_rmse": 0.78,
            },
        )
        _write_csv(
            out_dir / "cv_results.csv",
            "k,weights,mean_spearman,mean_rmse",
            ["1,uniform,0.40,0.80"],
        )
        _write_csv(
            out_dir / "test_predictions.csv",
            "word,gold,pred,error,abs_error",
            ["apple,4.6,4.4,-0.2,0.2"],
        )

    if include_prediction_full:
        _write_csv(out_dir / "oov_gold.csv", "word", ["missing_gold_word"])
        _write_csv(out_dir / "oov_vocab.csv", "word", ["missing_vocab_word"])
        if include_vocab_predictions:
            _write_csv(
                out_dir / "vocab_predictions.csv",
                "word,pred",
                ["apple,4.4"],
            )

    if include_holdout_summary:
        _write_json(
            out_dir / "holdout_summary.json",
            {
                "schema_version": "7.0.0",
                "holdout_spearman": 0.51,
            },
        )

    if include_holdout_unmatched:
        _write_csv(
            out_dir / "holdout_unmatched.csv",
            "source,word",
            ["holdout_only,justice"],
        )


def _write_vec(path: Path) -> None:
    vectors = {
        "apple": [0.95, 0.85, 0.10, 0.00, 0.00],
        "banana": [0.90, 0.80, 0.10, 0.10, 0.00],
        "table": [0.88, 0.90, 0.20, 0.10, 0.00],
        "river": [0.80, 0.70, 0.30, 0.10, 0.10],
        "castle": [0.84, 0.78, 0.25, 0.10, 0.10],
        "music": [0.30, 0.20, 0.88, 0.70, 0.30],
        "dream": [0.25, 0.20, 0.78, 0.82, 0.20],
        "idea": [0.10, 0.10, 0.95, 0.88, 0.55],
        "justice": [0.10, 0.00, 0.90, 0.92, 0.60],
        "theory": [0.12, 0.10, 0.86, 0.80, 0.50],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"{len(vectors)} 5\n")
        for word, values in vectors.items():
            handle.write(f"{word} {' '.join(str(v) for v in values)}\n")


def _base_config() -> dict:
    return {
        "dataset": {
            "gold": "data/gold.csv",
            "word_column": "Word",
            "score_column": "Conc.M",
            "lowercase": True,
            "pos_filter": {"enabled": False, "pos_column": "Dom_Pos", "tags": ["Noun"]},
        },
        "embeddings": {
            "mode": "single",
            "active_space": "mini",
            "spaces": [
                {
                    "id": "mini",
                    "kind": "vec",
                    "path": "embeddings/mini.vec",
                    "label": "mini_vec",
                }
            ],
        },
        "runtime": {
            "output_dir": "outputs/gui_run",
            "seed": 13,
            "n_jobs": 1,
        },
        "prediction": {
            "test_size": 0.25,
            "cv_folds": 3,
            "k_min": 1,
            "k_max": 3,
            "k_step": 1,
            "topn_neighbors": 3,
            "target": "data/vocab.txt",
            "holdout": "data/holdout.csv",
            "predictions_csv": "outputs/gui_run/vocab_predictions.csv",
            "prediction_column": "pred",
            "unmatched_csv": "outputs/gui_run/holdout_unmatched.csv",
        },
        "reports": {"level": "full"},
    }


class _MiniVecEmbeddings:
    def __init__(self, vec_path: str):
        lines = Path(vec_path).read_text(encoding="utf-8").strip().splitlines()
        header = lines[0].split()
        self.dim = int(header[1])
        self._vectors: dict[str, np.ndarray] = {}
        for line in lines[1:]:
            parts = line.split()
            word = parts[0]
            values = np.array([float(value) for value in parts[1:]], dtype=np.float32)
            self._vectors[word] = values

    def get_vector(self, word: str):
        return self._vectors.get(word)


def _patched_load_embeddings_by_space(embeddings_cfg: dict):
    out = {}
    for space in embeddings_cfg.get("spaces", []):
        if not isinstance(space, dict):
            continue
        space_id = str(space.get("id", "")).strip()
        path = str(space.get("path", "")).strip()
        if not space_id or not path:
            continue
        out[space_id] = _MiniVecEmbeddings(path)
    return out


def _wait_for_job(client, job_id: str, timeout_s: float = 30.0) -> dict:
    start = time.time()
    while time.time() - start < timeout_s:
        response = client.get(f"/api/jobs/{job_id}")
        assert response.status_code == 200
        payload = response.get_json()
        if payload["status"] in {"done", "error"}:
            return payload
        time.sleep(0.1)
    raise AssertionError(f"Timed out waiting for job {job_id}")


def _submit_prediction_run(client, config: dict, config_path: str = "configs/prediction.json") -> str:
    response = client.post(
        "/api/jobs/prediction-run",
        json={"config": json.loads(json.dumps(config)), "config_path": config_path},
    )
    assert response.status_code == 202
    return response.get_json()["job_id"]


def _submit_prediction_holdout(client, config: dict, config_path: str = "configs/prediction.json") -> str:
    response = client.post(
        "/api/jobs/prediction-holdout",
        json={"config": json.loads(json.dumps(config)), "config_path": config_path},
    )
    assert response.status_code == 202
    return response.get_json()["job_id"]


@pytest.fixture()
def gui_env(tmp_path: Path, monkeypatch):
    import concreteness_knn_core.prediction.prediction_data_prep as prep_mod

    monkeypatch.setattr(prep_mod, "load_embeddings_by_space", _patched_load_embeddings_by_space)

    _write_vec(tmp_path / "embeddings" / "mini.vec")
    _write_csv(
        tmp_path / "data" / "gold.csv",
        "Word,Conc.M",
        [
            "apple,4.6",
            "banana,4.3",
            "table,4.8",
            "river,4.2",
            "castle,4.5",
            "music,2.8",
            "dream,2.1",
            "idea,1.4",
        ],
    )
    _write_csv(
        tmp_path / "data" / "holdout.csv",
        "Word,Conc.M",
        [
            "apple,4.6",
            "justice,1.2",
            "idea,1.4",
            "table,4.8",
        ],
    )
    _write_csv(
        tmp_path / "data" / "holdout_no_overlap.csv",
        "Word,Conc.M",
        [
            "xword,2.0",
            "yword,3.0",
        ],
    )
    _write_vocab(tmp_path / "data" / "vocab.txt", ["apple", "justice", "idea", "table"])

    config = _base_config()
    _write_json(tmp_path / "configs" / "prediction.json", config)

    app = create_app(repo_root=tmp_path, testing=True)
    return {
        "root": tmp_path,
        "app": app,
        "client": app.test_client(),
        "config": config,
    }


def test_config_default_endpoint(gui_env):
    client = gui_env["client"]
    response = client.get("/api/config/default")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["source"] == "default_config"
    assert payload["raw_config"]["embeddings"]["mode"] == "single"


def test_index_page_contains_core_inputs_and_guided_ui(gui_env):
    client = gui_env["client"]
    response = client.get("/")
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "KNN-ConcrItUp" in html
    assert "Live-config control panel" in html
    assert "Repository Root" not in html
    assert "Workspace Controls" not in html
    assert "Config Tools" in html
    assert "Core Inputs" in html
    assert "Easy Entry" not in html
    assert "Config Editor" not in html
    assert 'id="tabGuided"' not in html
    assert 'id="tabRaw"' not in html
    assert 'id="rawJson"' not in html
    assert 'id="emb_mode"' not in html
    assert 'id="prediction_holdout"' not in html
    assert 'id="advancedConfig"' in html
    assert 'id="emb_single_kind"' in html
    assert 'id="embExistingSelect"' in html
    assert 'id="btnUploadGoldCsv"' in html
    assert 'id="btnUploadPredictVocab"' in html
    assert "Add gold file (CSV/TSV); set word/score columns." in html
    assert "Select embedding file or type a path." in html
    assert "Add targets file (one token per line)." in html

    core_idx = html.index("Core Inputs")
    adv_idx = html.index('id="advancedConfig"')
    gold_idx = html.index('id="dataset_gold"')
    word_idx = html.index('id="dataset_word_column"')
    score_idx = html.index('id="dataset_score_column"')
    existing_emb_idx = html.index('id="embExistingSelect"')
    emb_path_idx = html.index('id="emb_single_path"')
    emb_kind_idx = html.index('id="emb_single_kind"')
    target_idx = html.index('id="prediction_target"')
    assert core_idx < gold_idx < word_idx < score_idx < existing_emb_idx < emb_path_idx < emb_kind_idx < target_idx < adv_idx


def test_frontend_boot_autoloads_default_config_script_marker():
    js_path = Path(__file__).resolve().parents[1] / "static" / "app.js"
    script = js_path.read_text(encoding="utf-8")
    assert "await handleNewConfig();" in script
    assert 'headers.set("X-CSRF-Token", CSRF_TOKEN)' in script


def test_frontend_results_default_artifact_and_space_column_filter_markers():
    js_path = Path(__file__).resolve().parents[1] / "static" / "app.js"
    script = js_path.read_text(encoding="utf-8")
    assert "function pickDefaultArtifactKey(artifactKeys)" in script
    assert 'artifactKeys.includes("vocab_predictions")' in script
    assert 'key === "cv_results"' in script
    assert "allowPayloadError = false" in script
    assert "{ allowPayloadError: true }" in script
    assert 'if (job.status === "done")' in script
    assert 'else if (job.status === "error")' in script
    assert '"gold_coverage"' in script
    assert '"coverage_ratio"' not in script
    assert "function isSpaceSpecificColumn(columnName)" in script
    assert 'value.startsWith("pred__")' in script
    assert 'value.startsWith("neighbors__")' in script
    assert 'value.startsWith("neighbor_gold_scores__")' in script
    assert 'value.startsWith("neighbor_cosine_distances__")' in script


def test_config_load_save_upload_download(gui_env):
    client = gui_env["client"]

    save_response = client.post(
        "/api/config/save",
        json={
            "path": "configs/saved.json",
            "config": gui_env["config"],
        },
    )
    assert save_response.status_code == 200

    load_response = client.post("/api/config/load", json={"path": "configs/saved.json"})
    assert load_response.status_code == 200
    loaded = load_response.get_json()
    assert loaded["raw_config"]["dataset"]["gold"] == "data/gold.csv"
    assert loaded["existing_results"]["found"] is False
    assert loaded["existing_results"]["artifacts"] == {}

    upload_response = client.post(
        "/api/config/upload",
        data={
            "file": (
                io.BytesIO(json.dumps(gui_env["config"]).encode("utf-8")),
                "uploaded.json",
            )
        },
        content_type="multipart/form-data",
    )
    assert upload_response.status_code == 200
    uploaded = upload_response.get_json()
    assert uploaded["filename"] == "uploaded.json"
    assert uploaded["existing_results"]["found"] is False
    assert uploaded["existing_results"]["artifacts"] == {}

    download_response = client.get(
        "/api/config/download",
        query_string={
            "filename": "out.json",
            "config": json.dumps(gui_env["config"]),
        },
    )
    assert download_response.status_code == 200
    assert "attachment" in download_response.headers.get("Content-Disposition", "")


def test_config_load_auto_attaches_existing_results(gui_env):
    client = gui_env["client"]
    _write_existing_gui_artifacts(gui_env["root"])

    response = client.post("/api/config/load", json={"path": "configs/prediction.json"})
    assert response.status_code == 200
    payload = response.get_json()
    existing = payload["existing_results"]
    assert existing["found"] is True
    assert isinstance(existing["job_id"], str)
    assert "summary" in existing["artifacts"]
    assert "test_predictions" in existing["artifacts"]
    assert "holdout_summary" in existing["artifacts"]

    latest = client.get("/api/results/latest")
    assert latest.status_code == 200
    latest_job = latest.get_json()["latest_job"]
    assert latest_job is not None
    assert latest_job["job_id"] == existing["job_id"]
    assert "summary" in latest_job["artifacts"]


def test_config_upload_auto_attaches_existing_results(gui_env):
    client = gui_env["client"]
    _write_existing_gui_artifacts(gui_env["root"])

    response = client.post(
        "/api/config/upload",
        data={
            "file": (
                io.BytesIO(json.dumps(gui_env["config"]).encode("utf-8")),
                "uploaded.json",
            )
        },
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    payload = response.get_json()
    existing = payload["existing_results"]
    assert existing["found"] is True
    assert isinstance(existing["job_id"], str)
    assert "summary" in existing["artifacts"]

    latest = client.get("/api/results/latest")
    assert latest.status_code == 200
    latest_job = latest.get_json()["latest_job"]
    assert latest_job is not None
    assert latest_job["job_id"] == existing["job_id"]


def test_config_open_partial_existing_results_attaches_subset(gui_env):
    client = gui_env["client"]
    out_dir = gui_env["root"] / "outputs" / "gui_run"
    _write_json(out_dir / "summary.json", {"schema_version": "7.0.0"})

    response = client.post("/api/config/load", json={"path": "configs/prediction.json"})
    assert response.status_code == 200
    existing = response.get_json()["existing_results"]
    assert existing["found"] is True
    assert set(existing["artifacts"].keys()) == {"summary"}


def test_config_open_existing_results_path_policy_warning(gui_env):
    client = gui_env["client"]
    invalid_cfg = json.loads(json.dumps(gui_env["config"]))
    invalid_cfg["runtime"]["output_dir"] = "/tmp/outside-results"
    _write_json(gui_env["root"] / "configs" / "invalid_out_dir.json", invalid_cfg)

    response = client.post("/api/config/load", json={"path": "configs/invalid_out_dir.json"})
    assert response.status_code == 200
    existing = response.get_json()["existing_results"]
    assert existing["found"] is False
    assert existing["job_id"] is None
    assert existing["artifacts"] == {}
    assert existing["warnings"]
    assert "inside the repository root" in existing["warnings"][0]


def test_config_open_existing_results_deduplicates_job_registration(gui_env):
    client = gui_env["client"]
    _write_existing_gui_artifacts(gui_env["root"])

    first = client.post("/api/config/load", json={"path": "configs/prediction.json"})
    assert first.status_code == 200
    first_existing = first.get_json()["existing_results"]
    assert first_existing["found"] is True

    second = client.post("/api/config/load", json={"path": "configs/prediction.json"})
    assert second.status_code == 200
    second_existing = second.get_json()["existing_results"]
    assert second_existing["found"] is True
    assert second_existing["job_id"] == first_existing["job_id"]

    latest = client.get("/api/results/latest")
    assert latest.status_code == 200
    latest_job = latest.get_json()["latest_job"]
    assert latest_job is not None
    assert latest_job["job_id"] == first_existing["job_id"]


def test_fs_list_and_traversal_rejection(gui_env):
    client = gui_env["client"]

    listed = client.get("/api/fs/list", query_string={"path": "."})
    assert listed.status_code == 200
    payload = listed.get_json()
    assert isinstance(payload["entries"], list)

    rejected = client.get("/api/fs/list", query_string={"path": "../outside"})
    assert rejected.status_code == 400
    assert "inside the repository root" in rejected.get_json()["error"]


def test_fs_upload_saves_guided_files_with_collision_suffix(gui_env):
    client = gui_env["client"]

    first = client.post(
        "/api/fs/upload",
        data={
            "field_target": "gold",
            "file": (io.BytesIO(b"Word,Conc.M\napple,4.4\n"), "gold data.tsv"),
        },
        content_type="multipart/form-data",
    )
    assert first.status_code == 201
    first_payload = first.get_json()
    assert first_payload["field_target"] == "gold"
    assert first_payload["path"].startswith("data/uploads/gui/")
    first_path = gui_env["root"] / first_payload["path"]
    assert first_path.exists()
    assert "apple,4.4" in first_path.read_text(encoding="utf-8")

    second = client.post(
        "/api/fs/upload",
        data={
            "field_target": "gold",
            "file": (io.BytesIO(b"Word,Conc.M\nbanana,4.1\n"), "gold data.tsv"),
        },
        content_type="multipart/form-data",
    )
    assert second.status_code == 201
    second_payload = second.get_json()
    assert second_payload["path"] != first_payload["path"]
    second_path = gui_env["root"] / second_payload["path"]
    assert second_path.exists()

    bad_target = client.post(
        "/api/fs/upload",
        data={
            "field_target": "invalid",
            "file": (io.BytesIO(b"x"), "anything.txt"),
        },
        content_type="multipart/form-data",
    )
    assert bad_target.status_code == 400
    assert "field_target" in bad_target.get_json()["error"]


def test_fs_embeddings_lists_supported_files_with_filename_labels(gui_env):
    client = gui_env["client"]
    emb_root = gui_env["root"] / "embeddings"
    (emb_root / "nested").mkdir(parents=True, exist_ok=True)
    (emb_root / "nested" / "mini.vec").write_text("1 1\nx 1.0\n", encoding="utf-8")
    (emb_root / "ignore.bin").write_bytes(b"\x00\x01")
    (emb_root / "skip.mdl").write_text("ignore", encoding="utf-8")

    response = client.get("/api/fs/embeddings")
    assert response.status_code == 200
    payload = response.get_json()
    candidates = payload["candidates"]
    assert isinstance(candidates, list)
    assert any(item["path"].endswith("embeddings/mini.vec") for item in candidates)
    assert any(item["path"].endswith("embeddings/nested/mini.vec") for item in candidates)
    assert any(item["path"].endswith("embeddings/ignore.bin") for item in candidates)
    assert all(not item["path"].endswith("skip.mdl") for item in candidates)
    assert any("(nested)" in item["label"] for item in candidates if item["name"] == "mini.vec")
    assert all("/" not in item["label"] and "\\" not in item["label"] for item in candidates)


def test_prediction_run_and_holdout_live_config_jobs(gui_env):
    client = gui_env["client"]
    config = gui_env["config"]

    run_job_id = _submit_prediction_run(client, config)
    run_job = _wait_for_job(client, run_job_id)
    assert run_job["status"] == "done"
    artifacts = run_job["result"]["artifacts"]
    assert "summary" in artifacts
    assert (gui_env["root"] / artifacts["vocab_predictions"]).exists()

    holdout_config = json.loads(json.dumps(config))
    holdout_config["prediction"]["predictions_csv"] = artifacts["vocab_predictions"]
    holdout_job_id = _submit_prediction_holdout(client, holdout_config)
    holdout_job = _wait_for_job(client, holdout_job_id)
    assert holdout_job["status"] == "done"
    holdout_artifacts = holdout_job["result"]["artifacts"]
    assert (gui_env["root"] / holdout_artifacts["holdout_summary"]).exists()


def test_single_active_job_rejected(gui_env, monkeypatch):
    import apps.core_gui.app as app_mod

    client = gui_env["client"]
    config = gui_env["config"]

    original_run_prediction = app_mod.PredictionPipeline.run_prediction

    def slow_run_prediction(self, run_config, config_path):
        time.sleep(0.7)
        return original_run_prediction(self, run_config, config_path)

    monkeypatch.setattr(app_mod.PredictionPipeline, "run_prediction", slow_run_prediction)

    first = client.post(
        "/api/jobs/prediction-run",
        json={"config": config, "config_path": "configs/prediction.json"},
    )
    assert first.status_code == 202
    first_job_id = first.get_json()["job_id"]

    second = client.post(
        "/api/jobs/prediction-holdout",
        json={"config": config, "config_path": "configs/prediction.json"},
    )
    assert second.status_code == 409
    payload = second.get_json()
    assert payload["active_job"]["job_id"] == first_job_id
    assert payload["active_job"]["status"] in {"queued", "running"}

    done = _wait_for_job(client, first_job_id)
    assert done["status"] == "done"


def test_prediction_jobs_require_live_config(gui_env):
    client = gui_env["client"]

    run_missing = client.post(
        "/api/jobs/prediction-run",
        json={"config_path": "configs/prediction.json"},
    )
    assert run_missing.status_code == 400
    assert "live editor config" in run_missing.get_json()["error"]

    holdout_missing = client.post(
        "/api/jobs/prediction-holdout",
        json={"config_path": "configs/prediction.json"},
    )
    assert holdout_missing.status_code == 400
    assert "live editor config" in holdout_missing.get_json()["error"]


def test_results_latest_preview_and_download(gui_env):
    client = gui_env["client"]
    config = gui_env["config"]

    run_job_id = _submit_prediction_run(client, config)
    run_job = _wait_for_job(client, run_job_id)
    assert run_job["status"] == "done"

    latest = client.get("/api/results/latest")
    assert latest.status_code == 200
    latest_payload = latest.get_json()["latest_job"]
    assert latest_payload["job_id"] == run_job_id
    assert "test_predictions" in latest_payload["artifacts"]

    csv_preview = client.get(
        "/api/results/preview",
        query_string={
            "job_id": run_job_id,
            "artifact_key": "vocab_predictions",
            "q": "apple",
            "offset": 0,
            "limit": 10,
        },
    )
    assert csv_preview.status_code == 200
    csv_payload = csv_preview.get_json()
    assert csv_payload["kind"] == "csv"
    assert csv_payload["total_matches"] >= 1

    text_preview = client.get(
        "/api/results/preview",
        query_string={
            "job_id": run_job_id,
            "artifact_key": "summary",
            "q": "test_spearman",
            "offset": 0,
            "limit": 20,
        },
    )
    assert text_preview.status_code == 200
    text_payload = text_preview.get_json()
    assert text_payload["kind"] == "text"
    assert text_payload["total_matches"] >= 1

    download = client.get(
        "/api/results/download",
        query_string={
            "job_id": run_job_id,
            "artifact_key": "summary",
        },
    )
    assert download.status_code == 200
    assert "attachment" in download.headers.get("Content-Disposition", "")


def test_results_csv_preview_preserves_row_index_order(gui_env):
    client = gui_env["client"]
    out_dir = gui_env["root"] / "outputs" / "gui_run"
    _write_json(out_dir / "summary.json", {"schema_version": "7.0.0"})
    _write_csv(out_dir / "vocab_predictions.csv", "word,pred", ["Zulu,1.0", "alpha,2.0", "Beta,3.0"])

    response = client.post("/api/config/load", json={"path": "configs/prediction.json"})
    assert response.status_code == 200
    existing = response.get_json()["existing_results"]
    assert existing["found"] is True
    job_id = existing["job_id"]

    preview = client.get(
        "/api/results/preview",
        query_string={
            "job_id": job_id,
            "artifact_key": "vocab_predictions",
            "q": "",
            "offset": 0,
            "limit": 3,
        },
    )
    assert preview.status_code == 200
    rows = preview.get_json()["rows"]
    observed = [row["word"] for row in rows]
    row_indices = [row["_row_index"] for row in rows]
    assert observed == ["Zulu", "alpha", "Beta"]
    assert row_indices == ["0", "1", "2"]


def test_results_csv_search_is_word_only_and_disabled_for_cv(gui_env):
    client = gui_env["client"]
    out_dir = gui_env["root"] / "outputs" / "gui_run"
    _write_json(out_dir / "summary.json", {"schema_version": "7.0.0"})
    _write_csv(
        out_dir / "vocab_predictions.csv",
        "word,pred",
        [
            "alpha,apple",
            "beta,pear",
        ],
    )
    _write_csv(
        out_dir / "cv_results.csv",
        "k,weights,mean_spearman,mean_rmse",
        [
            "5,distance,0.80,0.50",
            "10,distance,0.82,0.48",
        ],
    )

    response = client.post("/api/config/load", json={"path": "configs/prediction.json"})
    assert response.status_code == 200
    existing = response.get_json()["existing_results"]
    assert existing["found"] is True
    job_id = existing["job_id"]

    non_word_hit = client.get(
        "/api/results/preview",
        query_string={
            "job_id": job_id,
            "artifact_key": "vocab_predictions",
            "q": "apple",
            "offset": 0,
            "limit": 10,
        },
    )
    assert non_word_hit.status_code == 200
    assert non_word_hit.get_json()["total_matches"] == 0

    word_hit = client.get(
        "/api/results/preview",
        query_string={
            "job_id": job_id,
            "artifact_key": "vocab_predictions",
            "q": "alpha",
            "offset": 0,
            "limit": 10,
        },
    )
    assert word_hit.status_code == 200
    word_payload = word_hit.get_json()
    assert word_payload["total_matches"] == 1
    assert word_payload["rows"][0]["word"] == "alpha"

    cv_query = client.get(
        "/api/results/preview",
        query_string={
            "job_id": job_id,
            "artifact_key": "cv_results",
            "q": "does_not_match_anything",
            "offset": 0,
            "limit": 10,
        },
    )
    assert cv_query.status_code == 200
    cv_payload = cv_query.get_json()
    assert cv_payload["query"] == ""
    assert cv_payload["total_matches"] == 2


def test_results_invalid_job_or_artifact(gui_env):
    client = gui_env["client"]
    config = gui_env["config"]

    bad_job = client.get(
        "/api/results/preview",
        query_string={
            "job_id": "missing-job",
            "artifact_key": "summary",
            "q": "",
            "offset": 0,
            "limit": 10,
        },
    )
    assert bad_job.status_code == 404

    run_job_id = _submit_prediction_run(client, config)
    run_job = _wait_for_job(client, run_job_id)
    assert run_job["status"] == "done"

    bad_artifact = client.get(
        "/api/results/preview",
        query_string={
            "job_id": run_job_id,
            "artifact_key": "unknown_artifact",
            "q": "",
            "offset": 0,
            "limit": 10,
        },
    )
    assert bad_artifact.status_code == 404


def test_prediction_job_error_paths(gui_env):
    client = gui_env["client"]

    invalid_response = client.post(
        "/api/jobs/prediction-run",
        json={"config": {"dataset": {"gold": ""}}, "config_path": "configs/prediction.json"},
    )
    assert invalid_response.status_code == 202
    invalid_job = _wait_for_job(client, invalid_response.get_json()["job_id"])
    assert invalid_job["status"] == "error"
    assert "dataset.gold" in invalid_job["error"]
    latest = client.get("/api/results/latest")
    assert latest.status_code == 200
    assert latest.get_json()["latest_job"] is None


def test_holdout_mismatch_error(gui_env):
    client = gui_env["client"]

    run_job_id = _submit_prediction_run(client, gui_env["config"])
    run_job = _wait_for_job(client, run_job_id)
    assert run_job["status"] == "done"

    mismatch_config = json.loads(json.dumps(gui_env["config"]))
    mismatch_config["prediction"]["holdout"] = "data/holdout_no_overlap.csv"
    mismatch_config["prediction"]["predictions_csv"] = run_job["result"]["artifacts"][
        "vocab_predictions"
    ]

    holdout_job_id = _submit_prediction_holdout(client, mismatch_config)
    holdout_job = _wait_for_job(client, holdout_job_id)
    assert holdout_job["status"] == "error"
    assert "No overlapping words" in holdout_job["error"]
