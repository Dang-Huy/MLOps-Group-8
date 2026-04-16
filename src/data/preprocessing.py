"""
src/data/preprocessing.py
=========================
Clean and type-cast the raw DataFrame before feature engineering.

Input  : raw DataFrame (post-validation) + optional outlier caps from train
Output : cleaned DataFrame + fitted caps dict (for valid/test consistency)
Role   : Wraps all cleaning logic proven in data_preparation.ipynb into
         reusable, cap-consistent functions.

Call order
----------
1. drop_pii_columns
2. clean_numeric_columns
3. cap_outliers          ← fit on train, apply to valid/test
4. parse_credit_history_age
5. clean_categorical_columns
6. encode_target         ← train/valid only
"""

import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.schema import (
    PII_COLUMNS, DIRTY_CATEGORICAL, AGE_RANGE,
    TARGET_COL, TARGET_ENCODING,
)


# ---------------------------------------------------------------------------
# 1. Drop PII
# ---------------------------------------------------------------------------

def drop_pii_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in PII_COLUMNS if c in df.columns]
    df = df.drop(columns=cols)
    print(f"[preprocessing] Dropped PII: {cols} — remaining cols: {df.shape[1]}")
    return df


# ---------------------------------------------------------------------------
# 2. Clean numeric columns
# ---------------------------------------------------------------------------

def clean_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip garbage chars, cast to numeric, enforce domain ranges."""
    df = df.copy()

    # Age
    df["Age"] = pd.to_numeric(
        df["Age"].astype(str).str.replace(r"[^\d]", "", regex=True),
        errors="coerce",
    )
    df["Age"] = df["Age"].where(df["Age"].between(*AGE_RANGE))

    # Annual_Income
    df["Annual_Income"] = pd.to_numeric(
        df["Annual_Income"].astype(str).str.replace("_", "", regex=False),
        errors="coerce",
    )

    # Num_of_Loan
    df["Num_of_Loan"] = pd.to_numeric(
        df["Num_of_Loan"].astype(str).str.replace("_", "", regex=False),
        errors="coerce",
    ).clip(lower=0, upper=10)

    # Num_of_Delayed_Payment
    df["Num_of_Delayed_Payment"] = pd.to_numeric(
        df["Num_of_Delayed_Payment"].astype(str).str.replace("_", "", regex=False),
        errors="coerce",
    ).clip(lower=0)

    # Outstanding_Debt
    df["Outstanding_Debt"] = pd.to_numeric(
        df["Outstanding_Debt"].astype(str).str.replace("_", "", regex=False),
        errors="coerce",
    )

    # Changed_Credit_Limit
    df["Changed_Credit_Limit"] = pd.to_numeric(
        df["Changed_Credit_Limit"].replace("_", np.nan),
        errors="coerce",
    )

    # Amount_invested_monthly
    df["Amount_invested_monthly"] = pd.to_numeric(
        df["Amount_invested_monthly"].replace("__10000__", np.nan),
        errors="coerce",
    )

    # Monthly_Balance
    df["Monthly_Balance"] = pd.to_numeric(
        df["Monthly_Balance"].replace("__-333333333333333333333333333__", np.nan),
        errors="coerce",
    )

    # Delay_from_due_date
    df["Delay_from_due_date"] = df["Delay_from_due_date"].clip(lower=0)

    return df


# ---------------------------------------------------------------------------
# 3. Cap outliers
# ---------------------------------------------------------------------------

def cap_outliers(
    df: pd.DataFrame,
    annual_income_cap: float | None = None,
) -> tuple[pd.DataFrame, float]:
    """
    Clip columns to valid domain ranges.
    Annual_Income cap is derived from train's 99th percentile and must be
    passed in for valid/test splits to avoid leakage.

    Returns (df, annual_income_cap)
    """
    df = df.copy()

    df["Num_Bank_Accounts"]    = df["Num_Bank_Accounts"].clip(0, 20)
    df["Num_Credit_Card"]      = df["Num_Credit_Card"].clip(0, 20)
    df["Interest_Rate"]        = df["Interest_Rate"].clip(1, 100)
    df["Num_Credit_Inquiries"] = df["Num_Credit_Inquiries"].clip(0, 20)

    if annual_income_cap is None:
        annual_income_cap = float(df["Annual_Income"].quantile(0.99))
        print(f"[preprocessing] Annual_Income 99th-pct cap = {annual_income_cap:,.2f}")

    df["Annual_Income"] = df["Annual_Income"].clip(upper=annual_income_cap)

    return df, annual_income_cap


# ---------------------------------------------------------------------------
# 4. Parse Credit_History_Age
# ---------------------------------------------------------------------------

def parse_credit_history_age(df: pd.DataFrame) -> pd.DataFrame:
    """Convert '22 Years and 9 Months' → 273 (total months)."""
    def _parse(val):
        if pd.isna(val):
            return np.nan
        m = re.search(r"(\d+)\s+Years?\s+and\s+(\d+)\s+Months?", str(val))
        return int(m.group(1)) * 12 + int(m.group(2)) if m else np.nan

    df = df.copy()
    df["Credit_History_Age"] = df["Credit_History_Age"].apply(_parse)
    return df


# ---------------------------------------------------------------------------
# 5. Clean categorical columns
# ---------------------------------------------------------------------------

def clean_categorical_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Replace known garbage placeholders with NaN for later imputation."""
    df = df.copy()
    df["Occupation"]              = df["Occupation"].replace("_______", np.nan)
    df["Credit_Mix"]              = df["Credit_Mix"].replace("_", np.nan)
    df["Payment_Behaviour"]       = df["Payment_Behaviour"].replace("!@9#%8", np.nan)
    df["Payment_of_Min_Amount"]   = df["Payment_of_Min_Amount"].replace("NM", np.nan)
    return df


# ---------------------------------------------------------------------------
# 6. Encode target
# ---------------------------------------------------------------------------

def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """Poor→0, Standard→1, Good→2. Skipped silently if column absent."""
    if TARGET_COL not in df.columns:
        return df
    df = df.copy()
    df[TARGET_COL] = df[TARGET_COL].map(TARGET_ENCODING)
    return df


# ---------------------------------------------------------------------------
# 7. Train/valid split  (only needed if starting from raw train CSV)
# ---------------------------------------------------------------------------

def split_train_valid(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified 80/20 split, reproducing the EDA notebook's split."""
    train, valid = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[TARGET_COL],
        shuffle=True,
    )
    return train.reset_index(drop=True), valid.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Public convenience — run full preprocessing on all three splits
# ---------------------------------------------------------------------------

def preprocess(
    train_raw: pd.DataFrame,
    valid_raw: pd.DataFrame,
    test_raw:  pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run the full preprocessing sequence on all three splits.
    Outlier caps are computed from train and applied to valid/test.

    Returns (train_clean, valid_clean, test_clean)
    """
    steps = [
        drop_pii_columns,
        clean_numeric_columns,
        parse_credit_history_age,
        clean_categorical_columns,
    ]

    def _apply(df):
        for fn in steps:
            df = fn(df)
        return df

    train = _apply(train_raw)
    valid = _apply(valid_raw)
    test  = _apply(test_raw)

    # Cap — fit on train, apply everywhere
    train, income_cap = cap_outliers(train)
    valid, _          = cap_outliers(valid, annual_income_cap=income_cap)
    test,  _          = cap_outliers(test,  annual_income_cap=income_cap)

    # Encode target (test has no target column — skipped automatically)
    train = encode_target(train)
    valid = encode_target(valid)

    print(f"[preprocessing] Done — train {train.shape}, valid {valid.shape}, test {test.shape}")
    return train, valid, test
