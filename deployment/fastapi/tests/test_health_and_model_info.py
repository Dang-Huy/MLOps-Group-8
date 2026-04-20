"""Minimal endpoint tests for deployment FastAPI app."""
from __future__ import annotations

import sys
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import deployment.fastapi.main as app_module


class FakeService:
    """Lightweight fake service for endpoint tests."""

    def __init__(self, source_resolved_from: str = "mlflow_stage_production") -> None:
        self.model_version = "1.0.0"
        self.uptime_seconds = 12.5
        self._source = source_resolved_from

    def get_model_info(self) -> dict:
        return {
            "model_name": "lightgbm",
            "model_version": "1.0.0",
            "model_source": "file:///D:/repo/artifacts/models/final_model_bundle.pkl",
            "run_id": "c7d625eebead4c8990ff726703e021b2",
            "alias_or_stage": "stage:Production",
            "metrics_core": {
                "final_test_f1_macro": 0.7862,
                "final_test_accuracy": 0.7930,
            },
            "best_params": {"n_estimators": 490},
            "params": {"final_model_name": "lightgbm"},
            "source_resolved_from": self._source,
            "warnings": [],
        }


@pytest.fixture
def client_mlflow_stage(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    fake = FakeService("mlflow_stage_production")
    monkeypatch.setattr(
        app_module.InferenceBackendService,
        "get_instance",
        classmethod(lambda cls, config=None: fake),
    )
    app_module._service = None
    with TestClient(app_module.app) as client:
        yield client
    app_module._service = None


def test_health_endpoint(client_mlflow_stage: TestClient) -> None:
    response = client_mlflow_stage.get("/health")
    assert response.status_code == 200

    payload = response.json()
    assert payload["status"] == "healthy"
    assert payload["model_loaded"] is True
    assert payload["model_version"] == "1.0.0"


def test_model_info_endpoint(client_mlflow_stage: TestClient) -> None:
    response = client_mlflow_stage.get("/model-info")
    assert response.status_code == 200

    payload = response.json()
    assert payload["model_name"] == "lightgbm"
    assert payload["source_resolved_from"] == "mlflow_stage_production"
    assert "metrics_core" in payload
    assert "best_params" in payload
