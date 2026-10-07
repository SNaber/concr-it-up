"""Regressions for config round trips, input ambiguity, search, and recovery."""

import copy
import io
import json
import unicodedata

import pytest

from apps.core_gui.app import create_app, _preview_csv, _preview_text
from apps.core_gui.hosted_config import HostedLimits, inspect_gold
from apps.core_gui.tests.test_hosted_app import _csrf, _post, HTTPS_ROOT
from concreteness_knn_core.config import validate_numeric_settings
from concreteness_knn_core.data import load_gold_df


@pytest.fixture()
def app(tmp_path, monkeypatch):
    models = []
    for identifier in ["arb_space", "de_space"]:
        model = tmp_path / (identifier + ".vec")
        model.write_text("1 2\nword 1 0\n")
        models.append({"id": identifier, "kind": "vec", "path": str(model), "label": identifier})
    for name, value in {
        "HOSTED_MODE": "1", "ACCESS_MODE": "anonymous", "SECRET_KEY": "test" * 16,
        "EMBEDDING_ALLOWLIST": json.dumps({"embeddings": models}),
        "JOB_DB": str(tmp_path / "state/jobs.sqlite3"), "JOB_ROOT": str(tmp_path / "jobs"),
        "SESSION_ROOT": str(tmp_path / "data/hosted_sessions"),
        "PAPER_CONFIG_ROOT": str(tmp_path / "configs/paper_runs"),
    }.items():
        monkeypatch.setenv("CONCRITUP_" + name, value)
    return create_app(repo_root=tmp_path, testing=True, start_embedded_worker=False)


def test_german_selection_survives_all_config_roundtrips(app):
    client = app.test_client()
    csrf = _csrf(client)
    cfg = client.get("/api/config/default", base_url=HTTPS_ROOT).json["raw_config"]
    # Also repair configs exported by the older interface with stale ids.
    cfg["embeddings"]["spaces"][0]["path"] = "embedding:de_space"
    saved = _post(client, "/api/config/save", csrf, json={"path": "german.json", "config": cfg})
    assert saved.status_code == 200
    loaded = _post(client, "/api/config/load", csrf, json={"path": saved.json["path"]})
    imported = _post(client, "/api/config/upload", csrf, data={
        "file": (io.BytesIO(json.dumps(cfg).encode()), "german.json")}, content_type="multipart/form-data")
    downloaded = client.get("/api/config/download", base_url=HTTPS_ROOT, query_string={"config": json.dumps(cfg)})
    for response in [saved, loaded, imported]:
        assert response.status_code == 200
        for key in ["raw_config", "resolved_config"]:
            model = response.json[key]["embeddings"]
            assert model["active_space"] == "de_space"
            assert model["spaces"][0]["id"] == "de_space"
            assert model["spaces"][0]["path"] == "embedding:de_space"
    assert downloaded.status_code == 200
    assert downloaded.json["embeddings"] == loaded.json["raw_config"]["embeddings"]
    assert str(app.config["GUI_REPO_ROOT"]) not in downloaded.text


def test_unknown_model_is_rejected_without_overwriting_config(app):
    client = app.test_client()
    csrf = _csrf(client)
    cfg = client.get("/api/config/default", base_url=HTTPS_ROOT).json["raw_config"]
    _post(client, "/api/config/save", csrf, json={"path": "saved.json", "config": cfg})
    invalid = copy.deepcopy(cfg)
    invalid["embeddings"]["spaces"][0]["path"] = "embedding:missing"
    response = _post(client, "/api/config/save", csrf, json={"path": "saved.json", "config": invalid})
    assert response.status_code == 400
    assert "not available" in response.json["error"]
    loaded = _post(client, "/api/config/load", csrf, json={"path": "saved.json"})
    assert loaded.json["raw_config"] == cfg


@pytest.mark.parametrize("text,message", [
    ("word,score\nBaum,5,1\nHaus,4,2\n", "line 2: expected 2 columns, found 3"),
    ("word,score\nBaum\n", "line 2: expected 2 columns, found 1"),
    ("word,score,score\nBaum,5,1\n", "unique"),
    ("word,\nBaum,5\n", "non-empty"),
    ('word,score\n"Baum,5\n', "Invalid gold table"),
])
def test_ambiguous_tables_rejected_by_both_readers(tmp_path, text, message):
    path = tmp_path / "gold.csv"
    path.write_text(text)
    cfg = {"gold": str(path), "word_column": "word", "score_column": "score"}
    for read in [lambda: load_gold_df(cfg), lambda: inspect_gold({"dataset": cfg}, HostedLimits())]:
        with pytest.raises(ValueError, match=message):
            read()


def test_quoted_fields_blank_lines_and_empty_cells_remain_supported(tmp_path):
    path = tmp_path / "gold.csv"
    path.write_text('\nword,score\n"a,b",2\n\n"a""b",3\n001,4\nNA,5\nmissing,\n', encoding="utf-16")
    cfg = {"gold": str(path), "word_column": "word", "score_column": "score", "lowercase": False}
    assert load_gold_df(cfg).set_index("word")["score"].to_dict() == {'a,b': 2, 'a"b': 3, '001': 4, 'NA': 5}
    assert inspect_gold({"dataset": cfg}, HostedLimits())["retained_rows"] == 4


def test_search_handles_canonical_unicode_and_preserves_distinctions(tmp_path):
    path = tmp_path / "predictions.csv"
    path.write_text("word,pred\nChicorée,5\nStraße,4\n001,3\n", encoding="utf-8")
    for form in ["NFC", "NFD"]:
        query = unicodedata.normalize(form, "CHICORÉE")
        csv_preview = _preview_csv(path, query, 0, 50)
        assert csv_preview["total_matches"] == 1
        assert csv_preview["rows"][0]["word"] == "Chicorée"
        assert _preview_text(path, query, 0, 50)["total_matches"] == 1
    assert _preview_csv(path, "strasse", 0, 50)["total_matches"] == 0
    assert _preview_csv(path, "001", 0, 50)["rows"][0]["word"] == "001"


@pytest.mark.parametrize("field,value", [
    ("k_min", 5.9), ("cv_folds", ""), ("cv_folds", True), ("topn_neighbors", None),
    ("test_size", float("nan")), ("test_size", True), ("test_size", 0), ("test_size", 1),
])
def test_invalid_numbers_rejected_before_saving_or_running(app, field, value):
    client = app.test_client()
    csrf = _csrf(client)
    config = {"prediction": {field: value}}
    with pytest.raises(ValueError):
        validate_numeric_settings(config)
    response = _post(client, "/api/config/save", csrf, json={"path": "invalid.json", "config": config})
    assert response.status_code == 400
    assert field in response.json["error"]
    listing = client.get("/api/fs/list?path=configs/user", base_url=HTTPS_ROOT).json
    assert not listing["entries"]


def test_active_job_recovery_is_scoped_to_owner_and_handles_completion(app):
    client = app.test_client()
    _csrf(client)
    with client.session_transaction() as session:
        owner = session["anonymous_owner"]
    store = app.extensions["concritup_job_store"]
    job = store.submit("prediction-run", {}, owner=owner, config={})
    response = client.get("/api/jobs/active", base_url=HTTPS_ROOT)
    assert response.json["active_job"]["job_id"] == job
    assert response.json["active_job"]["status"] == "queued"
    stranger = app.test_client()
    assert stranger.get("/api/jobs/active", base_url=HTTPS_ROOT).json["active_job"] is None
    store.fail(job, "Test finished")
    assert client.get("/api/jobs/active", base_url=HTTPS_ROOT).json["active_job"] is None


def test_disabled_pos_filter_does_not_reject_unused_pattern(tmp_path):
    path = tmp_path / "gold.csv"
    path.write_text("word,score\nBaum,5\n")
    cfg = {"dataset": {"gold": str(path), "word_column": "word", "score_column": "score",
                       "pos_filter": {"enabled": False, "token_pattern": "unused"}}}
    assert inspect_gold(cfg, HostedLimits())["retained_rows"] == 1
