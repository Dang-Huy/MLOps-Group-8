"""
src/features/imputers.py
========================
Fit and apply missing-value imputers.

Input  : DataFrame that may contain NaN (post-cleaning, pre-feature-engineering).
Output : DataFrame with no NaN in imputable columns + fitted imputer object.

Design rules
------------
* Imputation statistics are **computed from the training set only**.
* The fitted imputer must be serialised and reloaded at serving time.
* Strategy mirrors the notebook (data_preparation.ipynb Sec 7):
    - Numeric columns  -> median (robust to outliers)
    - Categorical cols -> mode
* `Type_of_Loan` is intentionally excluded -- it is dropped by the encoder.
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Union

import pandas as pd


# ---------------------------------------------------------------------------
# Column lists (kept in sync with data_preparation.ipynb)
# ---------------------------------------------------------------------------

NUMERIC_COLS: List[str] = [
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
    "Credit_History_Age",
    "Total_EMI_per_month",
    "Amount_invested_monthly",
    "Monthly_Balance",
]

CATEGORICAL_COLS: List[str] = [
    "Occupation",
    "Credit_Mix",
    "Payment_of_Min_Amount",
    "Payment_Behaviour",
]


# ---------------------------------------------------------------------------
# MedianImputer
# ---------------------------------------------------------------------------

class MedianImputer:
    """
    Impute numeric columns with their training-set median.

    Parameters
    ----------
    columns : list of str
        Columns to impute.  Columns absent from the DataFrame are silently skipped.
    """

    def __init__(self, columns: List[str] = NUMERIC_COLS):
        self.columns = columns
        self.medians_: Dict[str, float] = {}

    def fit(self, df: pd.DataFrame) -> "MedianImputer":
        self.medians_ = {
            col: float(df[col].median())
            for col in self.columns
            if col in df.columns
        }
        print("[MedianImputer] Fitted medians:")
        for col, val in self.medians_.items():
            print(f"  {col:<35}: {val:.4f}")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col, median_val in self.medians_.items():
            if col in df.columns:
                df[col] = df[col].fillna(median_val)
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    @property
    def statistics(self) -> Dict[str, float]:
        return dict(self.medians_)


# ---------------------------------------------------------------------------
# ModeImputer
# ---------------------------------------------------------------------------

class ModeImputer:
    """
    Impute categorical columns with their training-set mode.

    Parameters
    ----------
    columns : list of str
        Columns to impute.  Columns absent from the DataFrame are silently skipped.
    """

    def __init__(self, columns: List[str] = CATEGORICAL_COLS):
        self.columns = columns
        self.modes_: Dict[str, str] = {}

    def fit(self, df: pd.DataFrame) -> "ModeImputer":
        self.modes_ = {}
        for col in self.columns:
            if col in df.columns:
                mode_series = df[col].mode(dropna=True)
                self.modes_[col] = str(mode_series.iloc[0]) if len(mode_series) > 0 else ""
        print("[ModeImputer] Fitted modes:")
        for col, val in self.modes_.items():
            print(f"  {col:<35}: {val}")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col, mode_val in self.modes_.items():
            if col in df.columns:
                df[col] = df[col].fillna(mode_val)
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    @property
    def statistics(self) -> Dict[str, str]:
        return dict(self.modes_)


# ---------------------------------------------------------------------------
# ImputerPipeline  (combines both imputers in the correct order)
# ---------------------------------------------------------------------------

class ImputerPipeline:
    """
    Sequential pipeline: MedianImputer -> ModeImputer.

    Fitted state
    ------------
    numeric_imputer   : MedianImputer
    categorical_imputer : ModeImputer

    Usage
    -----
    >>> pipeline = ImputerPipeline()
    >>> train_df = pipeline.fit_transform(train_df)
    >>> valid_df = pipeline.transform(valid_df)
    >>> test_df  = pipeline.transform(test_df)
    """

    def __init__(
        self,
        numeric_cols: List[str] = NUMERIC_COLS,
        categorical_cols: List[str] = CATEGORICAL_COLS,
    ):
        self.numeric_imputer = MedianImputer(columns=numeric_cols)
        self.categorical_imputer = ModeImputer(columns=categorical_cols)

    def fit(self, df: pd.DataFrame) -> "ImputerPipeline":
        self.numeric_imputer.fit(df)
        self.categorical_imputer.fit(df)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.numeric_imputer.transform(df)
        df = self.categorical_imputer.transform(df)
        self._report_remaining_na(df)
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    # ------------------------------------------------------------------
    @staticmethod
    def _report_remaining_na(df: pd.DataFrame) -> None:
        remaining = df.isnull().sum()
        remaining = remaining[remaining > 0]
        if remaining.empty:
            print("[ImputerPipeline] OK - No missing values remaining.")
        else:
            print("[ImputerPipeline] WARN - Remaining NaN after imputation:")
            print(remaining.to_string())

    @property
    def statistics(self) -> Dict[str, Dict[str, Union[float, str]]]:
        return {
            "numeric":     self.numeric_imputer.statistics,
            "categorical": self.categorical_imputer.statistics,
        }


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def impute_missing(
    train_df: pd.DataFrame,
    *other_dfs: pd.DataFrame,
) -> Tuple[ImputerPipeline, pd.DataFrame, ...]:
    """
    Fit on train_df and transform all supplied DataFrames.

    Parameters
    ----------
    train_df   : Training DataFrame (statistics computed here).
    *other_dfs : Additional DataFrames (valid, test) to transform.

    Returns
    -------
    (pipeline, imputed_train, imputed_other_1, ...)

    Example
    -------
    >>> pipeline, train_imp, valid_imp, test_imp = impute_missing(
    ...     train_set, valid_set, test_set
    ... )
    """
    pipeline = ImputerPipeline()
    imputed_train = pipeline.fit_transform(train_df)
    imputed_others = tuple(pipeline.transform(df) for df in other_dfs)
    return (pipeline, imputed_train) + imputed_others


# ---------------------------------------------------------------------------
# CLI convenience
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import joblib
    import os

    parser = argparse.ArgumentParser(description="Impute missing values.")
    parser.add_argument("--train",   required=True)
    parser.add_argument("--valid",   required=False, default=None)
    parser.add_argument("--test",    required=False, default=None)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    train = pd.read_csv(args.train, low_memory=False)

    others, other_names = [], []
    for path, name in [(args.valid, "valid"), (args.test, "test")]:
        if path:
            others.append(pd.read_csv(path, low_memory=False))
            other_names.append(name)

    results = impute_missing(train, *others)
    pipeline, train_imp = results[0], results[1]
    rest = results[2:]

    train_imp.to_csv(f"{args.out_dir}/train_imputed.csv", index=False)
    for name, df in zip(other_names, rest):
        df.to_csv(f"{args.out_dir}/{name}_imputed.csv", index=False)

    joblib.dump(pipeline, f"{args.out_dir}/imputer_pipeline.pkl")
    print(f"Imputer pipeline saved to {args.out_dir}/imputer_pipeline.pkl")