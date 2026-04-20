"""Unit tests for src/data/preprocessing.py"""
# ruff: noqa: E402

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import pytest
from src.data.preprocessing import (
    drop_pii_columns, clean_numeric_columns,
    parse_credit_history_age, clean_categorical_columns,
    cap_outliers, encode_target,
)
from src.data.schema import TARGET_COL


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "ID": ["x1", "x2"],
        "Customer_ID": ["c1", "c2"],
        "Name": ["Alice", "Bob"],
        "SSN": ["111", "222"],
        "Month": ["Jan", "Feb"],
        "Age": ["25", "30_"],
        "Annual_Income": ["50000_", "60000"],
        "Monthly_Inhand_Salary": [4000.0, 5000.0],
        "Num_Bank_Accounts": [3, 5],
        "Num_Credit_Card": [2, 4],
        "Interest_Rate": [10, 15],
        "Num_of_Loan": ["3_", "4"],
        "Delay_from_due_date": [-1, 5],
        "Num_of_Delayed_Payment": ["5_", "3"],
        "Changed_Credit_Limit": ["_", "5.5"],
        "Num_Credit_Inquiries": [4.0, 6.0],
        "Credit_Mix": ["_", "Good"],
        "Outstanding_Debt": ["800_", "1200"],
        "Credit_Utilization_Ratio": [30.0, 45.0],
        "Credit_History_Age": ["22 Years and 3 Months", "NA"],
        "Payment_of_Min_Amount": ["Yes", "NM"],
        "Total_EMI_per_month": [50.0, 80.0],
        "Amount_invested_monthly": ["__10000__", "200.0"],
        "Payment_Behaviour": ["!@9#%8", "High_spent_Small_value_payments"],
        "Monthly_Balance": ["__-333333333333333333333333333__", "400.0"],
        TARGET_COL: ["Poor", "Good"],
    })


def test_drop_pii_columns(sample_df):
    result = drop_pii_columns(sample_df)
    pii_cols = ["ID", "Customer_ID", "Name", "SSN", "Month"]
    for col in pii_cols:
        assert col not in result.columns


def test_clean_numeric_age(sample_df):
    result = clean_numeric_columns(sample_df)
    assert result["Age"].iloc[0] == 25
    assert result["Age"].iloc[1] == 30


def test_clean_numeric_annual_income(sample_df):
    result = clean_numeric_columns(sample_df)
    assert result["Annual_Income"].iloc[0] == 50000.0


def test_parse_credit_history_age(sample_df):
    result = parse_credit_history_age(sample_df)
    assert result["Credit_History_Age"].iloc[0] == 22 * 12 + 3
    assert pd.isna(result["Credit_History_Age"].iloc[1])


def test_clean_categorical_columns(sample_df):
    result = clean_categorical_columns(sample_df)
    assert pd.isna(result["Credit_Mix"].iloc[0])           # "_" -> NaN
    assert pd.isna(result["Payment_of_Min_Amount"].iloc[1]) # "NM" -> NaN
    assert pd.isna(result["Payment_Behaviour"].iloc[0])     # "!@9#%8" -> NaN
    # Monthly_Balance garbage is cleaned in clean_numeric_columns, not here
    num_result = clean_numeric_columns(sample_df)
    assert pd.isna(num_result["Monthly_Balance"].iloc[0])


def test_cap_outliers(sample_df):
    df = clean_numeric_columns(sample_df)
    result, cap = cap_outliers(df)
    assert isinstance(cap, float)
    assert result["Num_Bank_Accounts"].max() <= 20


def test_encode_target(sample_df):
    result = encode_target(sample_df)
    assert result[TARGET_COL].tolist() == [0, 2]


def test_delay_clipped_to_zero(sample_df):
    result = clean_numeric_columns(sample_df)
    assert result["Delay_from_due_date"].iloc[0] >= 0
