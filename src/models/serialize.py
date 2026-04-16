"""
src/models/serialize.py
=======================
Save and load the complete model bundle atomically.
"""

import joblib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT    = Path(__file__).resolve().parent.parent.parent
ARTIFACT_DIR = REPO_ROOT / "artifacts" / "models"
BUNDLE_NAME  = "final_model_bundle.pkl"


@dataclass
class ModelBundle:
    model:    Any
    imputer:  Any
    encoder:  Any
    selector: Any
    metadata: dict = field(default_factory=dict)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the full preprocessing pipeline to a raw (or partly cleaned)
        DataFrame, returning the feature matrix ready for predict_proba().

        Mirrors the pipeline order used during training:
        preprocessing -> impute -> encode -> build_features -> select
        """
        from src.data.preprocessing import (
            drop_pii_columns, clean_numeric_columns,
            cap_outliers, parse_credit_history_age,
            clean_categorical_columns,
        )
        from src.features.build_features import build_features

        df = df.copy()

        # 1. Preprocessing
        for fn in [drop_pii_columns, clean_numeric_columns,
                   parse_credit_history_age, clean_categorical_columns]:
            df = fn(df)

        # Cap outliers -- use the cap stored in metadata if available
        income_cap = self.metadata.get("annual_income_cap")
        df, _ = cap_outliers(df, annual_income_cap=income_cap)

        # Drop target column if present (inference path)
        from src.data.schema import TARGET_COL
        if TARGET_COL in df.columns:
            df = df.drop(columns=[TARGET_COL])

        # 2. Impute
        df = self.imputer.transform(df)

        # 3. Encode
        df = self.encoder.transform(df)

        # 4. Build features
        df = build_features(df)

        # 5. Select
        feature_cols = [c for c in self.selector.feature_names_out if c in df.columns]
        df = df[feature_cols]

        return df


def save_bundle(bundle: ModelBundle, path: Path | None = None) -> Path:
    path = path or (ARTIFACT_DIR / BUNDLE_NAME)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path, compress=3)
    print(f"[serialize] Bundle saved -> {path}  ({path.stat().st_size / 1e6:.1f} MB)")
    return path


def load_bundle(path: Path | None = None) -> ModelBundle:
    path = path or (ARTIFACT_DIR / BUNDLE_NAME)
    if not path.exists():
        raise FileNotFoundError(f"No bundle at {path}. Run the training pipeline first.")
    bundle: ModelBundle = joblib.load(path)
    print(f"[serialize] Bundle loaded <- {path}")
    return bundle
