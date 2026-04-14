"""
src/data/validation.py
======================
Validate a raw DataFrame against the schema contract before preprocessing.

Input  : raw DataFrame from ingestion.py + schema from schema.py
Output : validation report dict; raises ValueError on critical failures
Role   : Quality gate — stops bad data entering the pipeline early
"""

import pandas as pd
from src.data.schema import RAW_SCHEMA, PII_COLUMNS, TARGET_COL


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
MAX_MISSING_RATE   = 0.40   # columns above this trigger a warning
CRITICAL_MISSING   = 0.80   # columns above this trigger an error
MIN_ROWS           = 1_000  # fewer rows than this is suspicious


def validate_raw(df: pd.DataFrame, split: str = "train") -> dict:
    """
    Run schema and quality checks on a raw DataFrame.

    Parameters
    ----------
    df    : Raw DataFrame from ingestion.py
    split : "train" | "valid" | "test" — affects whether target is expected

    Returns
    -------
    report : dict with keys:
        - passed          : bool
        - missing_columns : list of columns absent from df
        - extra_columns   : list of unexpected columns in df
        - high_missing    : dict {col: missing_pct} for cols > threshold
        - critical_missing: list of cols missing >80% of values
        - row_count       : int
        - errors          : list of blocking error messages
        - warnings        : list of non-blocking warning messages

    Raises
    ------
    ValueError if any critical check fails (missing required cols, too few rows,
                critical missing rate).
    """
    errors   = []
    warnings = []

    # ---- Row count -------------------------------------------------------
    if len(df) < MIN_ROWS:
        errors.append(f"Too few rows: {len(df)} (min {MIN_ROWS})")

    # ---- Column presence -------------------------------------------------
    expected = set(RAW_SCHEMA.keys())
    if split == "test":
        expected.discard(TARGET_COL)

    present         = set(df.columns)
    missing_columns = sorted(expected - present - set(PII_COLUMNS))  # PII may already be dropped
    extra_columns   = sorted(present - expected)

    if missing_columns:
        errors.append(f"Missing required columns: {missing_columns}")
    if extra_columns:
        warnings.append(f"Unexpected extra columns (will be ignored): {extra_columns}")

    # ---- Missing rate per column -----------------------------------------
    miss_rate    = df.isnull().mean()
    high_missing = {
        col: round(float(miss_rate[col]), 4)
        for col in df.columns
        if miss_rate[col] > MAX_MISSING_RATE
    }
    critical_cols = [col for col, rate in high_missing.items() if rate > CRITICAL_MISSING]

    if high_missing:
        warnings.append(f"High missing rate columns (>{MAX_MISSING_RATE*100:.0f}%): {high_missing}")
    if critical_cols:
        errors.append(f"Critical missing rate (>{CRITICAL_MISSING*100:.0f}%): {critical_cols}")

    # ---- Target presence (train/valid only) ------------------------------
    if split in ("train", "valid"):
        if TARGET_COL not in df.columns:
            errors.append(f"Target column '{TARGET_COL}' missing from {split} set")
        else:
            unexpected_labels = set(df[TARGET_COL].dropna().unique()) - {"Poor", "Standard", "Good"}
            if unexpected_labels:
                warnings.append(f"Unexpected target labels: {unexpected_labels}")

    # ---- Build report ----------------------------------------------------
    passed = len(errors) == 0
    report = {
        "passed":           passed,
        "split":            split,
        "row_count":        len(df),
        "col_count":        len(df.columns),
        "missing_columns":  missing_columns,
        "extra_columns":    extra_columns,
        "high_missing":     high_missing,
        "critical_missing": critical_cols,
        "errors":           errors,
        "warnings":         warnings,
    }

    # ---- Print summary ---------------------------------------------------
    status = "✅ PASSED" if passed else "❌ FAILED"
    print(f"[validation] {split} — {status}")
    for w in warnings:
        print(f"  ⚠️  {w}")
    for e in errors:
        print(f"  ❌ {e}")

    if not passed:
        raise ValueError(
            f"Validation failed for '{split}' split with {len(errors)} error(s):\n"
            + "\n".join(f"  - {e}" for e in errors)
        )

    return report