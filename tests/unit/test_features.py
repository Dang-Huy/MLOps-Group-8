"""Unit tests for src/features/"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import pytest
from src.features.build_features import build_features, DERIVED_FEATURES
from src.features.encoders import OrdinalEncoder, OneHotEncoder
from src.features.imputers import MedianImputer, ModeImputer


@pytest.fixture
def clean_df():
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "Age": np.random.randint(20, 60, n).astype(float),
        "Annual_Income": np.random.uniform(20000, 100000, n),
        "Monthly_Inhand_Salary": np.random.uniform(1500, 8000, n),
        "Num_Bank_Accounts": np.random.randint(1, 10, n).astype(float),
        "Num_Credit_Card": np.random.randint(1, 8, n).astype(float),
        "Interest_Rate": np.random.randint(5, 30, n).astype(float),
        "Num_of_Loan": np.random.randint(0, 6, n).astype(float),
        "Delay_from_due_date": np.random.randint(0, 30, n).astype(float),
        "Num_of_Delayed_Payment": np.random.randint(0, 20, n).astype(float),
        "Changed_Credit_Limit": np.random.uniform(-5, 20, n),
        "Num_Credit_Inquiries": np.random.randint(0, 10, n).astype(float),
        "Outstanding_Debt": np.random.uniform(100, 5000, n),
        "Credit_Utilization_Ratio": np.random.uniform(10, 80, n),
        "Credit_History_Age": np.random.randint(12, 360, n).astype(float),
        "Total_EMI_per_month": np.random.uniform(10, 200, n),
        "Amount_invested_monthly": np.random.uniform(10, 500, n),
        "Monthly_Balance": np.random.uniform(100, 1000, n),
        "Occupation": np.random.choice(["Engineer", "Lawyer", "Teacher"], n),
        "Credit_Mix": np.random.choice(["Good", "Standard", "Bad"], n),
        "Payment_of_Min_Amount": np.random.choice(["Yes", "No"], n),
        "Payment_Behaviour": np.random.choice(
            ["High_spent_Small_value_payments", "Low_spent_Small_value_payments"], n
        ),
    })


def test_build_features_adds_columns(clean_df):
    result = build_features(clean_df)
    for feat in DERIVED_FEATURES:
        assert feat in result.columns


def test_build_features_no_inf(clean_df):
    result = build_features(clean_df)
    derived_cols = list(DERIVED_FEATURES.keys())
    for col in derived_cols:
        assert not result[col].isin([float("inf"), float("-inf")]).any()


def test_ordinal_encoder():
    enc = OrdinalEncoder({"Bad": 0, "Standard": 1, "Good": 2})
    s = pd.Series(["Good", "Bad", "Standard", None])
    result = enc.fit_transform(s)
    assert result.iloc[0] == 2
    assert result.iloc[1] == 0
    assert pd.isna(result.iloc[3])


def test_ohe_fit_transform():
    enc = OneHotEncoder()
    s = pd.Series(["A", "B", "A", "C"], name="col")
    dummies = enc.fit_transform(s)
    assert set(dummies.columns) == {"col_A", "col_B", "col_C"}
    assert dummies.shape[0] == 4


def test_ohe_unseen_category():
    enc = OneHotEncoder()
    s_train = pd.Series(["A", "B"], name="col")
    enc.fit(s_train)
    s_test = pd.Series(["A", "C"], name="col")
    result = enc.transform(s_test)
    assert "col_C" not in result.columns
    assert "col_A" in result.columns


def test_median_imputer():
    imp = MedianImputer(["x"])
    df = pd.DataFrame({"x": [1.0, 2.0, np.nan, 4.0]})
    result = imp.fit_transform(df)
    assert not result["x"].isna().any()
    assert result["x"].iloc[2] == 2.0


def test_mode_imputer():
    imp = ModeImputer(["cat"])
    df = pd.DataFrame({"cat": ["A", "A", "B", None]})
    result = imp.fit_transform(df)
    assert not result["cat"].isna().any()
    assert result["cat"].iloc[3] == "A"
