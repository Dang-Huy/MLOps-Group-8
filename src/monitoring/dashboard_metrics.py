"""
src/monitoring/dashboard_metrics.py
=====================================
Format and export metrics in Power BI-ready tabular format.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np

from src.utils.io import write_csv

REPO_ROOT  = Path(__file__).resolve().parent.parent.parent
REPORT_DIR = REPO_ROOT / "artifacts" / "reports"


def build_performance_table(
    metrics: dict,
    model_version: str = "1.0.0",
    split: str = "validation",
) -> pd.DataFrame:
    """Convert a metrics dict to a flat Power BI table (one row per metric)."""
    ts   = datetime.utcnow().isoformat()
    rows = []
    for metric, value in metrics.items():
        if isinstance(value, (int, float)):
            rows.append({
                "timestamp":     ts,
                "model_version": model_version,
                "split":         split,
                "metric":        metric,
                "value":         round(float(value), 4),
            })
    return pd.DataFrame(rows)


def build_prediction_distribution_table(
    predictions: list | pd.Series | np.ndarray,
    model_version: str = "1.0.0",
    label: str = "batch",
) -> pd.DataFrame:
    """Build per-class prediction distribution for Power BI."""
    label_map = {0: "Poor", 1: "Standard", 2: "Good"}
    preds = [label_map.get(int(p), str(p)) for p in predictions]
    total = len(preds)
    ts    = datetime.utcnow().isoformat()

    from collections import Counter
    counts = Counter(preds)
    rows = []
    for cls in ["Poor", "Standard", "Good"]:
        cnt = counts.get(cls, 0)
        rows.append({
            "timestamp":     ts,
            "model_version": model_version,
            "label":         label,
            "credit_class":  cls,
            "count":         cnt,
            "percentage":    round(cnt / total * 100, 2) if total else 0,
        })
    return pd.DataFrame(rows)


def build_input_quality_table(monitor_result: dict) -> pd.DataFrame:
    """Flatten input monitor result into Power BI-ready table."""
    rows = []
    ts   = monitor_result.get("timestamp", datetime.utcnow().isoformat())

    for col, rate in monitor_result.get("missing_rates", {}).items():
        rows.append({
            "timestamp":  ts, "feature": col, "issue_type": "missing_rate",
            "value": rate, "threshold": 0.30, "alert": "YES" if rate > 0.30 else "NO",
        })

    for col, cnt in monitor_result.get("out_of_range", {}).items():
        rows.append({
            "timestamp":  ts, "feature": col, "issue_type": "out_of_range_count",
            "value": float(cnt), "threshold": 0.0, "alert": "YES",
        })

    return pd.DataFrame(rows)


def export_all_powerbi_tables(
    metrics: dict,
    predictions: list | np.ndarray,
    monitor_result: dict | None = None,
    model_version: str = "1.0.0",
    out_dir: str | Path | None = None,
) -> dict[str, Path]:
    """
    Export all Power BI tables to artifacts/reports/.

    Returns dict mapping table name -> saved file path.
    """
    out_dir = Path(out_dir or REPORT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = {}

    perf_df = build_performance_table(metrics, model_version=model_version)
    p = out_dir / "powerbi_model_performance.csv"
    write_csv(perf_df, p)
    saved["powerbi_model_performance"] = p

    dist_df = build_prediction_distribution_table(predictions, model_version=model_version)
    p = out_dir / "powerbi_prediction_distribution.csv"
    write_csv(dist_df, p)
    saved["powerbi_prediction_distribution"] = p

    if monitor_result:
        iq_df = build_input_quality_table(monitor_result)
        p = out_dir / "powerbi_input_quality.csv"
        write_csv(iq_df, p)
        saved["powerbi_input_quality"] = p

    print(f"[dashboard_metrics] Exported {len(saved)} Power BI tables -> {out_dir}")
    return saved
