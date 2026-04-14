"""API contract tests using TestClient."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import pytest

MODEL_PATH = ROOT / "artifacts" / "models" / "final_model_bundle.pkl"


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="model not trained yet")
def test_health_endpoint():
    from fastapi.testclient import TestClient
    import os
    os.environ["MODEL_PATH"] = str(MODEL_PATH)

    from src.serving.service import InferenceService
    InferenceService.reset()
    InferenceService.get_instance(MODEL_PATH)

    from src.serving.api import app, _service
    import src.serving.api as api_module
    api_module._service = InferenceService.get_instance(MODEL_PATH)

    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert "model_version" in data


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="model not trained yet")
def test_predict_endpoint_returns_valid_response():
    from fastapi.testclient import TestClient
    import src.serving.api as api_module
    from src.serving.service import InferenceService
    InferenceService.reset()
    api_module._service = InferenceService.get_instance(MODEL_PATH)

    client = TestClient(api_module.app)
    payload = {
        "Age": 30,
        "Annual_Income": 50000,
        "Monthly_Inhand_Salary": 4000,
        "Num_Bank_Accounts": 3,
        "Num_Credit_Card": 3,
        "Interest_Rate": 12,
        "Num_of_Loan": 2,
        "Delay_from_due_date": 5,
        "Num_of_Delayed_Payment": 3,
        "Changed_Credit_Limit": 5.0,
        "Num_Credit_Inquiries": 3,
        "Outstanding_Debt": 1000,
        "Credit_Utilization_Ratio": 35,
        "Credit_History_Age": 120,
        "Total_EMI_per_month": 60,
        "Amount_invested_monthly": 100,
        "Monthly_Balance": 300,
        "Occupation": "Engineer",
        "Credit_Mix": "Good",
        "Payment_of_Min_Amount": "Yes",
        "Payment_Behaviour": "Low_spent_Small_value_payments",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_class" in data
    assert data["predicted_class"] in ["Poor", "Standard", "Good"]
    assert "probabilities" in data
    assert "action" in data
    assert abs(sum(data["probabilities"].values()) - 1.0) < 0.01
