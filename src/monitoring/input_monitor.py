"""
src/monitoring/input_monitor.py
================================
Schema, missing-value, unseen-category, and range checks on incoming data.
Produces Power BI-friendly tabular output.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

from src.utils.io import write_csv

REPO_ROOT  = Path(__file__).resolve().parent.parent.parent
REPORT_DIR = REPO_ROOT / "artifacts" / "reports"

_RANGE_RULES: dict[str, tuple[float, float]] = {
    "Age":                  (18,  100),
    "Interest_Rate":        (1,   100),
    "Num_Bank_Accounts":    (0,    20),
    "Num_Credit_Card":      (0,    20),
    "Num_Credit_Inquiries": (0,    20),
    "Num_of_Loan":          (0,    10),
    "Delay_from_due_date":  (0,   999),
}

_CATEGORICAL_COLS = [
    "Occupation", "Credit_Mix", "Payment_of_Min_Amount", "Payment_Behaviour"
]


def monitor_input(
    df: pd.DataFrame,
    reference_stats: dict | None = None,
    label: str = "batch",
) -> dict:
    """
    Run input quality checks on a batch of records.

    Parameters
    ----------
    df              : incoming records (raw or pre-cleaned)
    reference_stats : dict with 'category_sets' from training data
    label           : batch identifier for the report

    Returns
    -------
    dict with schema_issues, missing_rates, out_of_range, unseen_categories
    """
    result = {
        "timestamp":          datetime.utcnow().isoformat(),
        "label":              label,
        "row_count":          len(df),
        "schema_issues":      [],
        "missing_rates":      {},
        "out_of_range":       {},
        "unseen_categories":  {},
    }

    # Missing rates per column
    for col in df.columns:
        rate = float(df[col].isna().mean())
        if rate > 0:
            result["missing_rates"][col] = round(rate, 4)

    # Range checks
    for col, (lo, hi) in _RANGE_RULES.items():
        if col in df.columns:
            numeric = pd.to_numeric(df[col], errors="coerce")
            violations = int(((numeric < lo) | (numeric > hi)).sum())
            if violations:
                result["out_of_range"][col] = violations

    # Unseen category check (vs reference if provided)
    if reference_stats and "category_sets" in reference_stats:
        for col in _CATEGORICAL_COLS:
            if col in df.columns and col in reference_stats["category_sets"]:
                known  = set(reference_stats["category_sets"][col])
                seen   = set(df[col].dropna().unique())
                unseen = seen - known
                if unseen:
                    result["unseen_categories"][col] = sorted(unseen)

    return result


def input_monitor_to_powerbi(monitor_result: dict) -> pd.DataFrame:
    """Flatten monitor_result into a Power BI-friendly table (one row per metric)."""
    rows = []
    ts  = monitor_result["timestamp"]
    lbl = monitor_result["label"]
    n   = monitor_result["row_count"]

    rows.append({"timestamp": ts, "label": lbl, "metric": "row_count",
                 "feature": "ALL", "value": n, "issue_type": "volume"})

    for col, rate in monitor_result["missing_rates"].items():
        rows.append({"timestamp": ts, "label": lbl, "metric": "missing_rate",
                     "feature": col, "value": rate, "issue_type": "missing"})

    for col, cnt in monitor_result["out_of_range"].items():
        rows.append({"timestamp": ts, "label": lbl, "metric": "out_of_range_count",
                     "feature": col, "value": float(cnt), "issue_type": "range_violation"})

    for col, cats in monitor_result["unseen_categories"].items():
        rows.append({"timestamp": ts, "label": lbl, "metric": "unseen_category_count",
                     "feature": col, "value": float(len(cats)), "issue_type": "unseen_category"})

    return pd.DataFrame(rows)


def save_input_monitor_report(
    monitor_result: dict,
    output_path: str | Path | None = None,
) -> Path:
    df = input_monitor_to_powerbi(monitor_result)
    path = Path(output_path or REPORT_DIR / "powerbi_input_quality.csv")
    write_csv(df, path)
    print(f"[input_monitor] Report saved -> {path}")
    return path
