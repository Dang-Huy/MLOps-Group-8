"""src/serving/schemas.py -- Pydantic v2 request/response schemas."""
from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    Age:                     Optional[float] = Field(None, description="Customer age in years")
    Annual_Income:           Optional[float] = Field(None, ge=0)
    Monthly_Inhand_Salary:   Optional[float] = Field(None, ge=0)
    Num_Bank_Accounts:       Optional[float] = Field(None, ge=0)
    Num_Credit_Card:         Optional[float] = Field(None, ge=0)
    Interest_Rate:           Optional[float] = Field(None, ge=0)
    Num_of_Loan:             Optional[float] = Field(None, ge=0)
    Delay_from_due_date:     Optional[float] = Field(None, ge=0)
    Num_of_Delayed_Payment:  Optional[float] = Field(None, ge=0)
    Changed_Credit_Limit:    Optional[float] = None
    Num_Credit_Inquiries:    Optional[float] = Field(None, ge=0)
    Outstanding_Debt:        Optional[float] = Field(None, ge=0)
    Credit_Utilization_Ratio: Optional[float] = None
    Credit_History_Age:      Optional[float] = Field(None, ge=0,
                                 description="Credit history age in months")
    Total_EMI_per_month:     Optional[float] = Field(None, ge=0)
    Amount_invested_monthly: Optional[float] = None
    Monthly_Balance:         Optional[float] = None
    Occupation:              Optional[str]   = None
    Credit_Mix:              Optional[str]   = Field(None, description="Bad / Standard / Good")
    Payment_of_Min_Amount:   Optional[str]   = Field(None, description="Yes / No")
    Payment_Behaviour:       Optional[str]   = None

    model_config = {"extra": "allow"}


class PredictResponse(BaseModel):
    predicted_class: str
    predicted_label: int
    probabilities:   dict[str, float]
    decision:        str
    action:          str
    confidence:      float
    model_version:   str


class BatchPredictRequest(BaseModel):
    records: list[PredictRequest]


class BatchPredictResponse(BaseModel):
    predictions:   list[PredictResponse]
    n_records:     int
    model_version: str


class HealthResponse(BaseModel):
    status:          str
    model_loaded:    bool
    model_version:   str
    uptime_seconds:  float
