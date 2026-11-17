from __future__ import annotations

import joblib
import pytest
from fastapi.testclient import TestClient

from intent_mlops.serve import create_app

PAYLOAD = {
    "administrative": 2,
    "administrative_duration": 60.5,
    "informational": 0,
    "informational_duration": 0.0,
    "product_related": 30,
    "product_related_duration": 1200.0,
    "bounce_rates": 0.02,
    "exit_rates": 0.03,
    "page_values": 8.5,
    "special_day": 0.0,
    "month": "Nov",
    "operating_systems": 2,
    "browser": 2,
    "region": 1,
    "traffic_type": 2,
    "visitor_type": "Returning_Visitor",
    "weekend": False,
}


@pytest.fixture()
def client(trained_pipeline, tmp_path):
    pipeline, _, _ = trained_pipeline
    model_file = tmp_path / "model.joblib"
    joblib.dump(pipeline, model_file)
    return TestClient(create_app(model_file))


def test_health_reports_loaded_model(client):
    body = client.get("/healthz").json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_returns_calibrated_probability(client):
    response = client.post("/predict", json=PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert 0 <= body["probability"] <= 1
    assert body["prediction"] in (0, 1)
    assert body["label"] == ("purchase" if body["prediction"] else "no_purchase")


def test_missing_model_degrades_gracefully(tmp_path):
    app = create_app(tmp_path / "does-not-exist.joblib")
    client = TestClient(app)
    assert client.get("/healthz").json()["status"] == "degraded"
    assert client.post("/predict", json=PAYLOAD).status_code == 503


@pytest.mark.parametrize(
    "field,value",
    [("month", "Jan"), ("bounce_rates", 1.5), ("administrative", -1), ("visitor_type", "Bot")],
)
def test_invalid_payload_is_rejected(client, field, value):
    payload = {**PAYLOAD, field: value}
    assert client.post("/predict", json=payload).status_code == 422


def test_model_info_lists_features(client):
    body = client.get("/model-info").json()
    assert "month" in body["features"]
    assert body["model_path"].endswith("model.joblib")
