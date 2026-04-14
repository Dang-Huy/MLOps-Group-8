"""
src/pipelines/validation_pipeline.py
======================================
Validate the trained model before deployment.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from src.models.serialize import load_bundle
from src.models.evaluate  import evaluate, save_report
from src.utils.logger     import get_logger
from src.utils.io         import write_json

logger    = get_logger(__name__)
THRESHOLDS = {"f1_macro": 0.70, "accuracy": 0.75}
REPORT_DIR = ROOT / "artifacts" / "reports"


def run_validation_pipeline(model_path: str | Path | None = None) -> dict:
    logger.info("== VALIDATION PIPELINE ==")

    bundle_path = Path(model_path or ROOT / "artifacts" / "models" / "final_model_bundle.pkl")
    bundle      = load_bundle(bundle_path)
    logger.info(f"Model v{bundle.metadata.get('model_version', '?')} loaded.")

    # Load test split from processed/
    import pandas as pd
    test_path = ROOT / "data" / "processed" / "test_split.csv"
    if not test_path.exists():
        logger.error("test_split.csv not found -- run training_pipeline first.")
        return {"passed": False, "reason": "test_split missing"}

    test_df = pd.read_csv(test_path)
    from src.data.schema import TARGET_COL
    X_test = test_df.drop(columns=[TARGET_COL], errors="ignore")
    y_test = test_df[TARGET_COL].astype(int).values

    # Transform using bundle pipeline
    from src.data.preprocessing import (
        drop_pii_columns, clean_numeric_columns, cap_outliers,
        parse_credit_history_age, clean_categorical_columns,
    )
    from src.features.build_features import build_features

    df = X_test.copy()
    for fn in [drop_pii_columns, clean_numeric_columns,
               parse_credit_history_age, clean_categorical_columns]:
        df = fn(df)
    df, _ = cap_outliers(df)
    df = bundle.imputer.transform(df)
    df = bundle.encoder.transform(df)
    df = build_features(df)
    feat_cols = [c for c in bundle.selector.feature_names_out if c in df.columns]
    X_t = df[feat_cols]

    y_prob = bundle.model.predict_proba(X_t)
    y_pred = np.argmax(y_prob, axis=1)

    report = evaluate(y_test, y_pred, y_prob, split="validation_gate")
    save_report(report, "eval_validation_gate.json")

    passed = all(report.get(k, 0) >= v for k, v in THRESHOLDS.items())
    result = {"passed": passed, "metrics": report, "thresholds": THRESHOLDS}
    write_json(result, REPORT_DIR / "validation_gate_result.json")
    logger.info(f"Validation gate: {'PASSED OK' if passed else 'FAILED FAIL'}")
    return result


if __name__ == "__main__":
    run_validation_pipeline()
