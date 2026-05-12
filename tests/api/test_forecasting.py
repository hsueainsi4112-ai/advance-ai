import random


_KNOWN_PRODUCT_ID = 1
_WINDOW_END = "2026-04-17"


def _sample_counts(seed: int = 42) -> list[int]:
    rng = random.Random(seed)
    return [rng.randint(0, 8) for _ in range(90)]


def test_forecast_known_product(client):
    response = client.post(
        "/api/v1/forecast",
        json={
            "instacart_product_id": _KNOWN_PRODUCT_ID,
            "daily_counts": _sample_counts(),
            "window_end_date": _WINDOW_END,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["product_id"] == _KNOWN_PRODUCT_ID
    assert body["input_window_days"] == 90
    assert body["output_horizon_days"] == 7
    assert body["model_version"] == "task1b_v1"
    assert len(body["forecast"]["predicted_counts"]) == 7
    assert len(body["forecast"]["dates"]) == 7
    assert body["forecast"]["dates"][0] == "2026-04-18"
    assert body["forecast"]["dates"][6] == "2026-04-24"
    for c in body["forecast"]["predicted_counts"]:
        assert isinstance(c, int)
        assert c >= 0
    assert len(body["shap"]["top_days"]) == 10
    day_indices = [d["day_index"] for d in body["shap"]["top_days"]]
    assert len(set(day_indices)) == 10
    for d in body["shap"]["top_days"]:
        assert 0 <= d["day_index"] < 90
        assert d["attribution"] >= 0.0


def test_forecast_unknown_product_is_404(client):
    response = client.post(
        "/api/v1/forecast",
        json={
            "instacart_product_id": 999999999,
            "daily_counts": _sample_counts(),
            "window_end_date": _WINDOW_END,
        },
    )
    assert response.status_code == 404


def test_forecast_wrong_window_length_is_400(client):
    response = client.post(
        "/api/v1/forecast",
        json={
            "instacart_product_id": _KNOWN_PRODUCT_ID,
            "daily_counts": [1] * 30,
            "window_end_date": _WINDOW_END,
        },
    )
    assert response.status_code == 400


def test_forecast_malformed_date_is_400(client):
    response = client.post(
        "/api/v1/forecast",
        json={
            "instacart_product_id": _KNOWN_PRODUCT_ID,
            "daily_counts": _sample_counts(),
            "window_end_date": "17-04-2026",
        },
    )
    assert response.status_code == 400


def test_forecast_missing_field_is_422(client):
    response = client.post(
        "/api/v1/forecast",
        json={
            "instacart_product_id": _KNOWN_PRODUCT_ID,
            "daily_counts": _sample_counts(),
        },
    )
    assert response.status_code == 422
