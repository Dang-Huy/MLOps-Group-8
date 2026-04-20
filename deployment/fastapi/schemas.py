"""Pydantic v2 schemas for deployment FastAPI backend."""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    Age: Optional[float] = Field(None, description="Customer age in years")
    Annual_Income: Optional[float] = Field(None, ge=0)
    Monthly_Inhand_Salary: Optional[float] = Field(None, ge=0)
    Num_Bank_Accounts: Optional[float] = Field(None, ge=0)
    Num_Credit_Card: Optional[float] = Field(None, ge=0)
    Interest_Rate: Optional[float] = Field(None, ge=0)
    Num_of_Loan: Optional[float] = Field(None, ge=0)
    Delay_from_due_date: Optional[float] = Field(None, ge=0)
    Num_of_Delayed_Payment: Optional[float] = Field(None, ge=0)
    Changed_Credit_Limit: Optional[float] = None
    Num_Credit_Inquiries: Optional[float] = Field(None, ge=0)
    Outstanding_Debt: Optional[float] = Field(None, ge=0)
    Credit_Utilization_Ratio: Optional[float] = None
    Credit_History_Age: Optional[float] = Field(None, ge=0, description="Credit history age in months")
    Total_EMI_per_month: Optional[float] = Field(None, ge=0)
    Amount_invested_monthly: Optional[float] = None
    Monthly_Balance: Optional[float] = None
    Occupation: Optional[str] = None
    Credit_Mix: Optional[str] = Field(None, description="Bad / Standard / Good")
    Payment_of_Min_Amount: Optional[str] = Field(None, description="Yes / No")
    Payment_Behaviour: Optional[str] = None

    model_config = {
        "extra": "allow",
        "json_schema_extra": {
            "example": {
                "Age": 34,
                "Annual_Income": 78000,
                "Monthly_Inhand_Salary": 5200,
                "Num_Bank_Accounts": 4,
                "Num_Credit_Card": 3,
                "Interest_Rate": 12,
                "Num_of_Loan": 2,
                "Delay_from_due_date": 3,
                "Num_of_Delayed_Payment": 1,
                "Changed_Credit_Limit": 8.5,
                "Num_Credit_Inquiries": 2,
                "Outstanding_Debt": 1200,
                "Credit_Utilization_Ratio": 24.3,
                "Credit_History_Age": 86,
                "Total_EMI_per_month": 350,
                "Amount_invested_monthly": 500,
                "Monthly_Balance": 1800,
                "Occupation": "Engineer",
                "Credit_Mix": "Good",
                "Payment_of_Min_Amount": "Yes",
                "Payment_Behaviour": "High_spent_Small_value_payments"
            }
        },
    }


class PredictResponse(BaseModel):
    predicted_class: str
    predicted_label: int
    probabilities: dict[str, float]
    decision: str
    action: str
    confidence: float
    model_version: str


class BatchPredictRequest(BaseModel):
    records: list[PredictRequest]

    model_config = {
        "json_schema_extra": {
            "example": {
                "records": [
                    {
                        "Age": 34,
                        "Annual_Income": 78000,
                        "Monthly_Inhand_Salary": 5200,
                        "Num_Bank_Accounts": 4,
                        "Num_Credit_Card": 3,
                        "Interest_Rate": 12,
                        "Num_of_Loan": 2,
                        "Delay_from_due_date": 3,
                        "Num_of_Delayed_Payment": 1,
                        "Changed_Credit_Limit": 8.5,
                        "Num_Credit_Inquiries": 2,
                        "Outstanding_Debt": 1200,
                        "Credit_Utilization_Ratio": 24.3,
                        "Credit_History_Age": 86,
                        "Total_EMI_per_month": 350,
                        "Amount_invested_monthly": 500,
                        "Monthly_Balance": 1800,
                        "Occupation": "Engineer",
                        "Credit_Mix": "Good",
                        "Payment_of_Min_Amount": "Yes",
                        "Payment_Behaviour": "High_spent_Small_value_payments"
                    }
                ]
            }
        }
    }


class BatchPredictResponse(BaseModel):
    predictions: list[PredictResponse]
    n_records: int
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
    uptime_seconds: float


class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    model_source: str
    run_id: str | None = None
    alias_or_stage: str | None = None
    metrics_core: dict[str, float] = Field(default_factory=dict)
    best_params: dict[str, Any] = Field(default_factory=dict)
    params: dict[str, str] = Field(default_factory=dict)
    source_resolved_from: Literal[
        "mlflow_alias_production",
        "mlflow_stage_production",
        "json_fallback",
    ]
    warnings: list[str] = Field(default_factory=list)
