from fastapi.testclient import TestClient

from credit_risk.api.app import app
from credit_risk.config import settings
from credit_risk.training.trainer import train


def test_health_and_predict(tmp_path, monkeypatch, sample_payload):
    dest = tmp_path / "model.joblib"
    train(use_synthetic=True, quick=True, output_path=dest)
    monkeypatch.setattr(settings, "model_path", dest)
    monkeypatch.setattr(settings, "api_key", "")

    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["model_loaded"] is True

        response = client.post("/v1/predict", json=sample_payload)
        assert response.status_code == 200
        body = response.json()
        assert "credit_score" in body
        assert "probability_of_default" in body
        assert body["risk_band"] in {"A", "B", "C", "D", "E"}


def test_predict_requires_api_key(tmp_path, monkeypatch, sample_payload):
    dest = tmp_path / "model.joblib"
    train(use_synthetic=True, quick=True, output_path=dest)
    monkeypatch.setattr(settings, "model_path", dest)
    monkeypatch.setattr(settings, "api_key", "uat-test-key")

    with TestClient(app) as client:
        denied = client.post("/v1/predict", json=sample_payload)
        assert denied.status_code == 401

        allowed = client.post(
            "/v1/predict",
            json=sample_payload,
            headers={"X-API-Key": "uat-test-key"},
        )
        assert allowed.status_code == 200
