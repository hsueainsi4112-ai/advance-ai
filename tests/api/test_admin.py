import os

from src.model_management import service as mm


def _admin_headers() -> dict:
    return {"X-Admin-Key": os.environ["AI_ADMIN_KEY"]}


def test_upload_model_without_admin_key_is_401(client):
    response = client.post(
        "/api/v1/admin/upload-model",
        data={"task": "task1a_knn", "version": "test_v2"},
        files={"file": ("dummy.joblib", b"stub", "application/octet-stream")},
    )
    assert response.status_code == 401


def test_upload_model_with_wrong_admin_key_is_401(client):
    response = client.post(
        "/api/v1/admin/upload-model",
        headers={"X-Admin-Key": "wrong-key"},
        data={"task": "task1a_knn", "version": "test_v2"},
        files={"file": ("dummy.joblib", b"stub", "application/octet-stream")},
    )
    assert response.status_code == 401


def test_upload_model_rejects_unknown_task(client):
    response = client.post(
        "/api/v1/admin/upload-model",
        headers=_admin_headers(),
        data={"task": "bogus_task", "version": "x"},
        files={"file": ("dummy.bin", b"stub", "application/octet-stream")},
    )
    assert response.status_code == 400


def test_upload_model_positive_updates_version(client):
    version_tag = "admin_test_v99"
    original = mm.version_for("task1a_knn")
    try:
        response = client.post(
            "/api/v1/admin/upload-model",
            headers=_admin_headers(),
            data={"task": "task1a_knn", "version": version_tag},
            files={"file": ("dummy.joblib", b"stub-bytes", "application/octet-stream")},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "success"
        assert body["task"] == "task1a_knn"
        assert body["version"] == version_tag
        assert mm.version_for("task1a_knn") == version_tag

        follow_up = client.get("/api/v1/recommendations/1")
        assert follow_up.status_code == 200
        assert follow_up.json()["model_version"] == version_tag
    finally:
        mm._active_versions["task1a_knn"] = original


def test_readiness_endpoint_reports_active_versions(client):
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert set(body["models_loaded"].keys()) == {
        "task1a_knn",
        "task1b_lstm",
        "task2_resnet",
        "task2_mobilenet",
    }
