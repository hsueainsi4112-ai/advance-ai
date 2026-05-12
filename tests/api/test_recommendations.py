def test_recommendations_known_user(client):
    response = client.get("/api/v1/recommendations/1")
    assert response.status_code == 200
    body = response.json()
    assert body["user_id"] == 1
    assert body["model_version"] == "task1a_v1"
    assert len(body["reorder"]) == 10
    assert len(body["new_for_you"]) == 10
    reorder_ids = {r["product_id"] for r in body["reorder"]}
    new_ids = {r["product_id"] for r in body["new_for_you"]}
    assert reorder_ids.isdisjoint(new_ids)
    for rec in body["reorder"]:
        assert rec["bucket"] == "reorder"
        assert rec["name"]
        assert rec["explanation"]
        assert isinstance(rec["product_id"], int)
        assert isinstance(rec["score"], float)
    for rec in body["new_for_you"]:
        assert rec["bucket"] == "new_for_you"
        assert rec["explanation"]


def test_recommendations_unknown_user_falls_back_to_popularity(client):
    response = client.get("/api/v1/recommendations/999999999")
    assert response.status_code == 200
    body = response.json()
    assert body["user_id"] == 999999999
    assert body["reorder"] == []
    assert len(body["new_for_you"]) == 10
    for rec in body["new_for_you"]:
        assert rec["bucket"] == "new_for_you"
        assert rec["explanation"] == "Popular across the platform"


def test_recommendations_invalid_user_id_is_422(client):
    response = client.get("/api/v1/recommendations/not-an-int")
    assert response.status_code == 422
