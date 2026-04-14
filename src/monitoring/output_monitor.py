"""
src/monitoring/output_monitor.py
=================================
Monitor prediction and confidence distribution over time.
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
CLASS_LABELS = ["Poor", "Standard", "Good"]


def monitor_output(
    predictions: np.ndarray | list,
    probabilities: np.ndarray | None = None,
    label: str = "batch",
) -> dict:
    """
    Analyse prediction and confidence distribution.

    Parameters
    ----------
    predictions   : 1-D array of predicted integer labels (0/1/2)
    probabilities : (N, 3) probability matrix -- optional
    label         : batch/window identifier

    Returns
    -------
    dict with distribution, percentages, and confidence statistics
    """
    preds = np.asarray(predictions)
    result = {
        "timestamp":               datetime.utcnow().isoformat(),
        "label":                   label,
        "n_predictions":           int(len(preds)),
        "prediction_distribution": {},
        "prediction_pct":          {},
    }

    for k, cls in enumerate(CLASS_LABELS):
        cnt = int((preds == k).sum())
        result["prediction_distribution"][cls] = cnt
        result["prediction_pct"][cls] = round(cnt / len(preds) * 100, 2) if len(preds) else 0

    if probabilities is not None:
        proba    = np.asarray(probabilities)
        max_conf = proba.max(axis=1)
        result["mean_confidence"]  = round(float(max_conf.mean()), 4)
        result["p10_confidence"]   = round(float(np.percentile(max_conf, 10)), 4)
        result["p25_confidence"]   = round(float(np.percentile(max_conf, 25)), 4)
        result["p75_confidence"]   = round(float(np.percentile(max_conf, 75)), 4)
        result["p90_confidence"]   = round(float(np.percentile(max_conf, 90)), 4)
        result["low_conf_pct"]     = round(float((max_conf < 0.60).mean() * 100), 2)

    return result


def output_monitor_to_powerbi(monitor_result: dict) -> pd.DataFrame:
    """Flatten output monitor into a per-class row Power BI table."""
    rows = []
    ts  = monitor_result["timestamp"]
    lbl = monitor_result["label"]

    for cls in CLASS_LABELS:
        cnt = monitor_result["prediction_distribution"].get(cls, 0)
        pct = monitor_result["prediction_pct"].get(cls, 0)
        rows.append({
            "timestamp":    ts,
            "label":        lbl,
            "credit_class": cls,
            "count":        cnt,
            "percentage":   pct,
        })
    return pd.DataFrame(rows)


def save_output_monitor_report(
    monitor_result: dict,
    output_path: str | Path | None = None,
) -> Path:
    df   = output_monitor_to_powerbi(monitor_result)
    path = Path(output_path or REPORT_DIR / "powerbi_prediction_distribution.csv")
    write_csv(df, path)
    print(f"[output_monitor] Report saved -> {path}")
    return path
