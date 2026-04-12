"""
src/features/build_features.py
================================
Build derived features from the preprocessed DataFrame.

Input  : Preprocessed DataFrame (output of src/data/preprocessing.py)
Output : DataFrame with additional engineered feature columns appended.

All transformations are deterministic and stateless — no fitting required.
The same function must be called identically at training time and serving time
to avoid training-serving skew.
"""

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Feature-building helpers
# ---------------------------------------------------------------------------

def _debt_to_income(df: pd.DataFrame) -> pd.Series:
    """Monthly outstanding debt burden relative to monthly income."""
    monthly_income = df["Annual_Income"] / 12
    return df["Outstanding_Debt"] / monthly_income.replace(0, np.nan)


def _loan_to_income(df: pd.DataFrame) -> pd.Series:
    """Number of active loans scaled by annual income bucket."""
    return df["Num_of_Loan"] / (df["Annual_Income"] / 10_000).replace(0, np.nan)


def _emi_to_income(df: pd.DataFrame) -> pd.Series:
    """Total monthly EMI as a fraction of monthly in-hand salary."""
    return df["Total_EMI_per_month"] / df["Monthly_Inhand_Salary"].replace(0, np.nan)


def _savings_rate(df: pd.DataFrame) -> pd.Series:
    """Amount invested monthly as a fraction of monthly in-hand salary."""
    return df["Amount_invested_monthly"] / df["Monthly_Inhand_Salary"].replace(0, np.nan)


def _balance_to_income(df: pd.DataFrame) -> pd.Series:
    """Ratio of monthly balance to monthly in-hand salary."""
    return df["Monthly_Balance"] / df["Monthly_Inhand_Salary"].replace(0, np.nan)


def _credit_card_utilization(df: pd.DataFrame) -> pd.Series:
    """
    Proxy credit-card utilization: outstanding debt per credit card.
    Distinct from the raw Credit_Utilization_Ratio column.
    """
    return df["Outstanding_Debt"] / df["Num_Credit_Card"].replace(0, np.nan)


def _delay_per_loan(df: pd.DataFrame) -> pd.Series:
    """Average delay from due date normalised by number of active loans."""
    return df["Delay_from_due_date"] / df["Num_of_Loan"].replace(0, np.nan)


def _delinquency_rate(df: pd.DataFrame) -> pd.Series:
    """Fraction of payments that were delayed."""
    total_payments = df["Num_of_Loan"] * 12  # rough proxy
    return df["Num_of_Delayed_Payment"] / total_payments.replace(0, np.nan)


def _inquiry_per_account(df: pd.DataFrame) -> pd.Series:
    """Credit inquiries per bank account — signals credit-seeking behaviour."""
    return df["Num_Credit_Inquiries"] / df["Num_Bank_Accounts"].replace(0, np.nan)


def _credit_age_years(df: pd.DataFrame) -> pd.Series:
    """Credit history age expressed in years (rounded to 1 decimal)."""
    return (df["Credit_History_Age"] / 12).round(1)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

DERIVED_FEATURES = {
    "debt_to_income":        _debt_to_income,
    "loan_to_income":        _loan_to_income,
    "emi_to_income":         _emi_to_income,
    "savings_rate":          _savings_rate,
    "balance_to_income":     _balance_to_income,
    "credit_card_util_proxy": _credit_card_utilization,
    "delay_per_loan":        _delay_per_loan,
    "delinquency_rate":      _delinquency_rate,
    "inquiry_per_account":   _inquiry_per_account,
    "credit_age_years":      _credit_age_years,
}


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Append engineered features to the preprocessed DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed DataFrame — output of src/data/preprocessing.py.
        Expected columns include: Annual_Income, Outstanding_Debt,
        Monthly_Inhand_Salary, Num_of_Loan, Total_EMI_per_month,
        Amount_invested_monthly, Monthly_Balance, Num_Credit_Card,
        Delay_from_due_date, Num_of_Delayed_Payment, Num_Credit_Inquiries,
        Num_Bank_Accounts, Credit_History_Age.

    Returns
    -------
    pd.DataFrame
        Original DataFrame with additional derived feature columns appended.
        Infinite values produced by division are replaced with NaN.
    """
    df = df.copy()

    for feature_name, feature_fn in DERIVED_FEATURES.items():
        values = feature_fn(df)
        # Replace ±inf (from divide-by-zero when denominator is non-NaN zero)
        df[feature_name] = values.replace([np.inf, -np.inf], np.nan)

    print(f"[build_features] Added {len(DERIVED_FEATURES)} derived features.")
    print(f"  New columns: {list(DERIVED_FEATURES.keys())}")
    print(f"  Output shape: {df.shape}")

    return df


# ---------------------------------------------------------------------------
# CLI convenience
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build engineered features.")
    parser.add_argument("--input",  required=True, help="Path to preprocessed CSV.")
    parser.add_argument("--output", required=True, help="Path to save feature CSV.")
    args = parser.parse_args()

    raw = pd.read_csv(args.input, low_memory=False)
    out = build_features(raw)
    out.to_csv(args.output, index=False)
    print(f"Saved to {args.output}")