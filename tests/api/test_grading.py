import base64
import io

from PIL import Image

_USER_HEADERS = {"X-User-Id": "test-grade-user"}


def test_grade_image_returns_valid_grade(client, sample_jpeg_bytes, class_names):
    response = client.post(
        "/api/v1/grade-image",
        files={"file": ("sample.jpg", sample_jpeg_bytes, "image/jpeg")},
        headers=_USER_HEADERS,
    )
    assert response.status_code == 200
    body = response.json()
    assert body["grade"] in {"A", "B", "C"}
    assert body["grad_cam_url"].startswith("/api/v1/grad-cam/")
    assert body["predicted_class"] in class_names
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["model_version"] == "task2_v1"
    weights = body["ensemble_weights"]
    assert abs(weights["resnet50"] + weights["mobilenetv2"] - 1.0) < 1e-6
    expected_action = {
        "A": "sell_full_price",
        "B": "sell_discounted",
        "C": "flag_for_surplus",
    }[body["grade"]]
    assert body["recommended_action"] == expected_action


def test_grad_cam_roundtrip_after_grade(client, sample_jpeg_bytes):
    post_response = client.post(
        "/api/v1/grade-image",
        files={"file": ("sample.jpg", sample_jpeg_bytes, "image/jpeg")},
        headers=_USER_HEADERS,
    )
    request_id = post_response.json()["grad_cam_url"].rsplit("/", 1)[1]
    get_response = client.get(f"/api/v1/grad-cam/{request_id}")
    assert get_response.status_code == 200
    body = get_response.json()
    decoded = base64.b64decode(body["heatmap_base64"])
    with Image.open(io.BytesIO(decoded)) as heat:
        assert heat.size == (224, 224)
    assert body["predicted_class"]
    assert body["explanation"]
    assert body["model_version"] == "task2_v1"


def test_grade_image_rejects_wrong_mime(client):
    response = client.post(
        "/api/v1/grade-image",
        files={"file": ("bad.txt", b"not an image", "text/plain")},
        headers=_USER_HEADERS,
    )
    assert response.status_code == 400


def test_grade_image_rejects_oversized_payload(client):
    blob = b"\xff\xd8\xff\xe0" + b"x" * (11 * 1024 * 1024)
    response = client.post(
        "/api/v1/grade-image",
        files={"file": ("big.jpg", blob, "image/jpeg")},
        headers=_USER_HEADERS,
    )
    assert response.status_code == 413


def test_grade_image_rejects_oversized_dimensions(client, oversized_jpeg_bytes):
    response = client.post(
        "/api/v1/grade-image",
        files={"file": ("huge.jpg", oversized_jpeg_bytes, "image/jpeg")},
        headers=_USER_HEADERS,
    )
    assert response.status_code == 400


def test_grade_image_missing_user_id_is_400(client, sample_jpeg_bytes):
    response = client.post(
        "/api/v1/grade-image",
        files={"file": ("sample.jpg", sample_jpeg_bytes, "image/jpeg")},
    )
    assert response.status_code == 400
    assert response.json()["code"] == "missing_user_id"


def test_grad_cam_unknown_request_id(client):
    response = client.get("/api/v1/grad-cam/00000000-0000-0000-0000-000000000000")
    assert response.status_code == 404


def test_grade_image_rate_limit_returns_429(client, sample_jpeg_bytes):
    headers = {"X-User-Id": "rate-test-user"}
    for _ in range(10):
        response = client.post(
            "/api/v1/grade-image",
            files={"file": ("sample.jpg", sample_jpeg_bytes, "image/jpeg")},
            headers=headers,
        )
        assert response.status_code == 200
    response = client.post(
        "/api/v1/grade-image",
        files={"file": ("sample.jpg", sample_jpeg_bytes, "image/jpeg")},
        headers=headers,
    )
    assert response.status_code == 429
