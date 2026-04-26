"""
src/features/encoders.py
========================
Fit and apply categorical encoders.

Input  : Raw/preprocessed DataFrame with categorical columns.
Output : Encoded DataFrame + fitted encoder objects (for serialisation).

Design rules
------------
* All encoders are **fit on the training set only**.
* The fitted objects must be serialised (via src/models/serialize.py) and
  reloaded at serving time so that validation and production splits receive
  exactly the same transformation.
* Mirrors the encoding logic in notebooks/data_preparation.ipynb so that
  the same mappings apply across train, valid, test, and live inference.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import pandas as pd

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Ordinal mappings (fixed domain knowledge -- not learned from data)
# ---------------------------------------------------------------------------

CREDIT_MIX_MAP: Dict[str, int] = {"Bad": 0, "Standard": 1, "Good": 2}
PAYMENT_MIN_MAP: Dict[str, int] = {"No": 0, "Yes": 1}

# Columns for one-hot encoding
OHE_COLUMNS: List[str] = ["Occupation", "Payment_Behaviour"]


# ---------------------------------------------------------------------------
# OrdinalEncoder
# ---------------------------------------------------------------------------

class OrdinalEncoder:
    """
    Map string categories to integer ordinals using a fixed dictionary.

    Parameters
    ----------
    mapping : dict
        {category_string: integer_ordinal}
    handle_unknown : int or None
        Value assigned to unseen categories at transform time.
        None -> NaN (float column).
    """

    def __init__(self, mapping: Dict[str, int], handle_unknown: Optional[int] = None):
        self.mapping = mapping
        self.handle_unknown = handle_unknown
        self._is_fitted = False

    # fit is a no-op -- mapping is fixed by domain knowledge
    def fit(self, series: pd.Series) -> "OrdinalEncoder":
        self._is_fitted = True
        return self

    def transform(self, series: pd.Series) -> pd.Series:
        result = series.map(self.mapping)
        if self.handle_unknown is not None:
            result = result.fillna(self.handle_unknown)
        return result

    def fit_transform(self, series: pd.Series) -> pd.Series:
        return self.fit(series).transform(series)


# ---------------------------------------------------------------------------
# OneHotEncoder
# ---------------------------------------------------------------------------

class OneHotEncoder:
    """
    One-hot encode a single categorical column.

    fit() learns the set of valid categories from the training data.
    transform() creates indicator columns for each known category and fills 0
    for any unseen category encountered at serving time.

    Parameters
    ----------
    drop_first : bool
        Whether to drop the first dummy column (avoids multicollinearity).
        Default False to retain interpretability for all categories.
    """

    def __init__(self, drop_first: bool = False):
        self.drop_first = drop_first
        self.categories_: Optional[List[str]] = None
        self.column_name_: Optional[str] = None

    def fit(self, series: pd.Series) -> "OneHotEncoder":
        self.column_name_ = series.name
        categories = sorted(series.dropna().unique().tolist())
        if self.drop_first and len(categories) > 1:
            categories = categories[1:]
        self.categories_ = categories
        return self

    def transform(self, series: pd.Series) -> pd.DataFrame:
        if self.categories_ is None:
            raise RuntimeError("Call fit() before transform().")
        dummies = pd.get_dummies(series, prefix=self.column_name_, drop_first=False)
        expected_cols = [f"{self.column_name_}_{cat}" for cat in self.categories_]
        # Add missing columns (unseen categories at serving time -> all zeros)
        for col in expected_cols:
            if col not in dummies.columns:
                dummies[col] = 0
        # Drop extra columns not seen during training (new categories at serving)
        dummies = dummies[expected_cols]
        return dummies.astype(bool)

    def fit_transform(self, series: pd.Series) -> pd.DataFrame:
        return self.fit(series).transform(series)

    @property
    def feature_names_out(self) -> List[str]:
        if self.categories_ is None:
            raise RuntimeError("Encoder not yet fitted.")
        return [f"{self.column_name_}_{cat}" for cat in self.categories_]


# ---------------------------------------------------------------------------
# CategoricalEncoderPipeline  (thin wrapper over the above)
# ---------------------------------------------------------------------------

class CategoricalEncoderPipeline:
    """
    Orchestrates all categorical encodings for the credit-score dataset.

    Fitted state
    ------------
    ordinal_encoders : dict  {col_name: OrdinalEncoder}
    ohe_encoders     : dict  {col_name: OneHotEncoder}
    drop_columns     : list  columns removed before encoding (e.g. Type_of_Loan)
    train_columns    : list  final ordered column list after encoding (set at fit time)
    """

    def __init__(self, drop_columns: Optional[List[str]] = None):
        self.drop_columns: List[str] = drop_columns or ["Type_of_Loan"]
        self.ordinal_encoders: Dict[str, OrdinalEncoder] = {}
        self.ohe_encoders: Dict[str, OneHotEncoder] = {}
        self.train_columns: Optional[List[str]] = None

    # ------------------------------------------------------------------
    def fit(self, df: pd.DataFrame) -> "CategoricalEncoderPipeline":
        """
        Fit all encoders on the training DataFrame.

        Parameters
        ----------
        df : pd.DataFrame  -- training split (before encoding).

        Returns
        -------
        self
        """
        # Ordinal encoders (fixed mappings -- fit is a no-op but kept for API symmetry)
        self.ordinal_encoders["Credit_Mix"] = OrdinalEncoder(CREDIT_MIX_MAP).fit(
            df.get("Credit_Mix", pd.Series(dtype=str))
        )
        self.ordinal_encoders["Payment_of_Min_Amount"] = OrdinalEncoder(PAYMENT_MIN_MAP).fit(
            df.get("Payment_of_Min_Amount", pd.Series(dtype=str))
        )

        # One-hot encoders -- learn categories from train
        for col in OHE_COLUMNS:
            if col in df.columns:
                enc = OneHotEncoder(drop_first=False)
                enc.fit(df[col])
                self.ohe_encoders[col] = enc

        # Store expected output columns
        self.train_columns = self._encode(df).columns.tolist()
        print(f"[CategoricalEncoderPipeline] Fitted on {len(df)} rows.")
        return self

    # ------------------------------------------------------------------
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply fitted encoders to any split.

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        pd.DataFrame  Encoded DataFrame aligned to training columns.
        """
        encoded = self._encode(df)

        if self.train_columns is not None:
            # Align: add missing cols with 0, drop extra cols
            for col in self.train_columns:
                if col not in encoded.columns:
                    encoded[col] = 0
            # Keep only columns that exist in encoded (target col may be absent in test)
            cols_to_keep = [c for c in self.train_columns if c in encoded.columns]
            encoded = encoded[cols_to_keep]

        return encoded

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    # ------------------------------------------------------------------
    def _encode(self, df: pd.DataFrame) -> pd.DataFrame:
        """Internal: applies all transformations without column alignment."""
        df = df.copy()

        # Drop non-informative columns
        cols_to_drop = [c for c in self.drop_columns if c in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)

        # Ordinal
        for col, enc in self.ordinal_encoders.items():
            if col in df.columns:
                df[col] = enc.transform(df[col])

        # One-hot
        ohe_frames = []
        for col, enc in self.ohe_encoders.items():
            if col in df.columns:
                ohe_frames.append(enc.transform(df[col]))
                df = df.drop(columns=[col])

        if ohe_frames:
            df = pd.concat([df] + ohe_frames, axis=1)

        return df

    # ------------------------------------------------------------------
    @property
    def feature_names_out(self) -> List[str]:
        if self.train_columns is None:
            raise RuntimeError("Pipeline not yet fitted.")
        return self.train_columns


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def encode_features(
    train_df: pd.DataFrame,
    *other_dfs: pd.DataFrame,
) -> Tuple[CategoricalEncoderPipeline, pd.DataFrame, ...]:
    """
    Fit on train_df and transform all supplied DataFrames.

    Parameters
    ----------
    train_df   : Training DataFrame (encoder is fit here).
    *other_dfs : Additional DataFrames (valid, test) to transform.

    Returns
    -------
    (pipeline, encoded_train, encoded_other_1, encoded_other_2, ...)

    Example
    -------
    >>> pipeline, train_enc, valid_enc, test_enc = encode_features(
    ...     train_set, valid_set, test_set
    ... )
    """
    pipeline = CategoricalEncoderPipeline()
    encoded_train = pipeline.fit_transform(train_df)
    encoded_others = tuple(pipeline.transform(df) for df in other_dfs)
    return (pipeline, encoded_train) + encoded_others


# ---------------------------------------------------------------------------
# CLI convenience
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import joblib
    import os

    parser = argparse.ArgumentParser(description="Encode categorical features.")
    parser.add_argument("--train",   required=True)
    parser.add_argument("--valid",   required=False, default=None)
    parser.add_argument("--test",    required=False, default=None)
    parser.add_argument("--out-dir", required=True, help="Directory for encoded CSVs + encoder pkl.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    train = pd.read_csv(args.train, low_memory=False)

    others = []
    other_names = []
    for path, name in [(args.valid, "valid"), (args.test, "test")]:
        if path:
            others.append(pd.read_csv(path, low_memory=False))
            other_names.append(name)

    results = encode_features(train, *others)
    pipeline, train_enc = results[0], results[1]
    rest = results[2:]

    train_enc.to_csv(f"{args.out_dir}/train_encoded.csv", index=False)
    for name, df in zip(other_names, rest):
        df.to_csv(f"{args.out_dir}/{name}_encoded.csv", index=False)

    joblib.dump(pipeline, f"{args.out_dir}/encoder_pipeline.pkl")
    print(f"Encoder pipeline saved to {args.out_dir}/encoder_pipeline.pkl")