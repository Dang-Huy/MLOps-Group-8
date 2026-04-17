"""
src/features/selectors.py
==========================
Select the final feature subset used for model training and serving.

Input  : Fully encoded and imputed DataFrame (post-encoders + post-imputers).
Output : DataFrame containing only the selected feature columns (+ target if present).

Design rules
------------
* Feature selection logic is **determined at training time** and frozen.
* The fitted selector is serialised (via src/models/serialize.py) and applied
  identically at serving time to guarantee column alignment.
* Three selection strategies are provided:
    1. ExplicitSelector  -- hand-picked list from domain knowledge / notebook EDA.
    2. VarianceSelector  -- drops near-zero-variance columns (statistical).
    3. FeatureSelector   -- composable pipeline wrapping both.
* The default selection (SELECTED_FEATURES) mirrors the stable feature set
  observed in the data_preparation notebook after encoding.
"""

from __future__ import annotations

from typing import List, Optional, Set

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Default hand-picked feature list
# ---------------------------------------------------------------------------
# Derived from: EDA notebook, domain knowledge of credit scoring, and
# exclusion of columns that are PII, ID-like, or dropped during encoding.

SELECTED_FEATURES: List[str] = [
    # --- Raw numeric ---
    "Age",
    "Annual_Income",
    "Monthly_Inhand_Salary",
    "Num_Bank_Accounts",
    "Num_Credit_Card",
    "Interest_Rate",
    "Num_of_Loan",
    "Delay_from_due_date",
    "Num_of_Delayed_Payment",
    "Changed_Credit_Limit",
    "Num_Credit_Inquiries",
    "Outstanding_Debt",
    "Credit_Utilization_Ratio",
    "Credit_History_Age",   # = Credit_History_Months in EDA; parsed to total months
    "Total_EMI_per_month",
    "Amount_invested_monthly",
    "Monthly_Balance",
    # --- Encoded ordinal ---
    "Credit_Mix",
    "Payment_of_Min_Amount",
    # --- Derived features (from build_features.py) ---
    "debt_to_income",
    "loan_to_income",
    "emi_to_income",
    "savings_rate",
    "balance_to_income",
    "credit_card_util_proxy",
    "delay_per_loan",
    "delinquency_rate",
    "inquiry_per_account",
    "credit_age_years",
    # --- One-hot: Occupation ---
    "Occupation_Accountant",
    "Occupation_Architect",
    "Occupation_Developer",
    "Occupation_Doctor",
    "Occupation_Engineer",
    "Occupation_Entrepreneur",
    "Occupation_Journalist",
    "Occupation_Lawyer",
    "Occupation_Manager",
    "Occupation_Mechanic",
    "Occupation_Media_Manager",
    "Occupation_Musician",
    "Occupation_Scientist",
    "Occupation_Teacher",
    "Occupation_Writer",
    # --- One-hot: Payment_Behaviour ---
    "Payment_Behaviour_High_spent_Large_value_payments",
    "Payment_Behaviour_High_spent_Medium_value_payments",
    "Payment_Behaviour_High_spent_Small_value_payments",
    "Payment_Behaviour_Low_spent_Large_value_payments",
    "Payment_Behaviour_Low_spent_Medium_value_payments",
    "Payment_Behaviour_Low_spent_Small_value_payments",
]

TARGET_COL: str = "Credit_Score"


# ---------------------------------------------------------------------------
# ExplicitSelector
# ---------------------------------------------------------------------------

class ExplicitSelector:
    """
    Keep only a pre-specified list of feature columns.

    Parameters
    ----------
    features : list of str
        Columns to retain.  Columns absent in the DataFrame are silently skipped.
    target_col : str or None
        If present in the DataFrame, this column is always preserved regardless
        of whether it appears in `features`.
    """

    def __init__(
        self,
        features: List[str] = SELECTED_FEATURES,
        target_col: Optional[str] = TARGET_COL,
    ):
        self.features = list(features)
        self.target_col = target_col
        self.selected_features_: Optional[List[str]] = None   # set at fit time

    def fit(self, df: pd.DataFrame) -> "ExplicitSelector":
        present = [f for f in self.features if f in df.columns]
        missing = [f for f in self.features if f not in df.columns]
        if missing:
            print(f"[ExplicitSelector] [WARN]  {len(missing)} requested features not found "
                  f"in DataFrame and will be skipped: {missing}")
        self.selected_features_ = present
        print(f"[ExplicitSelector] Selected {len(present)} features.")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.selected_features_ is None:
            raise RuntimeError("Call fit() before transform().")
        cols = list(self.selected_features_)
        if self.target_col and self.target_col in df.columns:
            cols = cols + [self.target_col]
        # Fill any columns that were in training but are absent here (e.g. new OHE column)
        for col in cols:
            if col not in df.columns:
                df = df.copy()
                df[col] = 0
        return df[cols]

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    @property
    def feature_names_out(self) -> List[str]:
        if self.selected_features_ is None:
            raise RuntimeError("Selector not yet fitted.")
        return list(self.selected_features_)


# ---------------------------------------------------------------------------
# VarianceSelector
# ---------------------------------------------------------------------------

class VarianceSelector:
    """
    Drop columns whose variance on the training set is below `threshold`.

    Useful for removing near-constant one-hot columns that add no signal.

    Parameters
    ----------
    threshold : float
        Minimum variance to retain a column.  Default 0.01.
    exclude   : set of str
        Columns to never drop (e.g. the target).
    """

    def __init__(self, threshold: float = 0.01, exclude: Optional[Set[str]] = None):
        self.threshold = threshold
        self.exclude: Set[str] = exclude or {TARGET_COL}
        self.keep_cols_: Optional[List[str]] = None

    def fit(self, df: pd.DataFrame) -> "VarianceSelector":
        numeric_df = df.select_dtypes(include=[np.number])
        variances = numeric_df.var()
        low_var = variances[variances < self.threshold].index.tolist()
        low_var = [c for c in low_var if c not in self.exclude]
        self.keep_cols_ = [c for c in df.columns if c not in low_var]
        if low_var:
            print(f"[VarianceSelector] Dropping {len(low_var)} low-variance columns: {low_var}")
        else:
            print("[VarianceSelector] No low-variance columns found.")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.keep_cols_ is None:
            raise RuntimeError("Call fit() before transform().")
        cols = [c for c in self.keep_cols_ if c in df.columns]
        return df[cols]

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    @property
    def dropped_features(self) -> List[str]:
        if self.keep_cols_ is None:
            raise RuntimeError("Selector not yet fitted.")
        return []   # populated only after fit; use self.keep_cols_ directly


# ---------------------------------------------------------------------------
# FeatureSelector  (composable pipeline)
# ---------------------------------------------------------------------------

class FeatureSelector:
    """
    Composable selector pipeline.

    Steps (applied in order at transform time):
    1. ExplicitSelector -- keep hand-picked columns.
    2. VarianceSelector -- additionally drop near-constant columns.

    Parameters
    ----------
    features        : explicit feature list passed to ExplicitSelector.
    variance_threshold : float, minimum variance; set to 0 to disable.
    target_col      : target column name (always preserved).
    """

    def __init__(
        self,
        features: List[str] = SELECTED_FEATURES,
        variance_threshold: float = 0.01,
        target_col: Optional[str] = TARGET_COL,
    ):
        self.explicit = ExplicitSelector(features=features, target_col=target_col)
        self.variance = VarianceSelector(
            threshold=variance_threshold,
            exclude={target_col} if target_col else set(),
        )
        self.target_col = target_col
        self.final_features_: Optional[List[str]] = None

    def fit(self, df: pd.DataFrame) -> "FeatureSelector":
        after_explicit = self.explicit.fit_transform(df)
        self.variance.fit(after_explicit)
        after_variance = self.variance.transform(after_explicit)
        feature_cols = [c for c in after_variance.columns if c != self.target_col]
        self.final_features_ = feature_cols
        print(f"[FeatureSelector] Final feature count: {len(feature_cols)}")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.explicit.transform(df)
        df = self.variance.transform(df)
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    @property
    def feature_names_out(self) -> List[str]:
        if self.final_features_ is None:
            raise RuntimeError("Selector not yet fitted.")
        return list(self.final_features_)


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def select_features(
    train_df: pd.DataFrame,
    *other_dfs: pd.DataFrame,
    features: List[str] = SELECTED_FEATURES,
    variance_threshold: float = 0.01,
    target_col: Optional[str] = TARGET_COL,
) -> tuple:
    """
    Fit on train_df and select features in all supplied DataFrames.

    Returns
    -------
    (selector, selected_train, selected_other_1, ...)

    Example
    -------
    >>> selector, X_train, X_valid, X_test = select_features(
    ...     train_enc, valid_enc, test_enc
    ... )
    """
    selector = FeatureSelector(
        features=features,
        variance_threshold=variance_threshold,
        target_col=target_col,
    )
    selected_train = selector.fit_transform(train_df)
    selected_others = tuple(selector.transform(df) for df in other_dfs)
    return (selector, selected_train) + selected_others


# ---------------------------------------------------------------------------
# CLI convenience
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse, joblib, os

    parser = argparse.ArgumentParser(description="Select final feature subset.")
    parser.add_argument("--train",   required=True)
    parser.add_argument("--valid",   required=False, default=None)
    parser.add_argument("--test",    required=False, default=None)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--variance-threshold", type=float, default=0.01,
        help="Drop columns with variance below this value (default: 0.01)."
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    train = pd.read_csv(args.train, low_memory=False)

    others, other_names = [], []
    for path, name in [(args.valid, "valid"), (args.test, "test")]:
        if path:
            others.append(pd.read_csv(path, low_memory=False))
            other_names.append(name)

    results = select_features(train, *others, variance_threshold=args.variance_threshold)
    selector, train_sel = results[0], results[1]
    rest = results[2:]

    train_sel.to_csv(f"{args.out_dir}/train_selected.csv", index=False)
    for name, df in zip(other_names, rest):
        df.to_csv(f"{args.out_dir}/{name}_selected.csv", index=False)

    joblib.dump(selector, f"{args.out_dir}/feature_selector.pkl")
    print(f"Feature selector saved to {args.out_dir}/feature_selector.pkl")
    print(f"Final features ({len(selector.feature_names_out)}): {selector.feature_names_out}")