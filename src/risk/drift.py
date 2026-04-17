"""
src/risk/drift.py
=================
Drift detection and summary logic.
Wraps PSI checks and produces structured drift reports.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

from src.risk.psi import compute_psi, psi_summary
from src.utils.io import write_csv, write_json

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DRIFT_DIR = REPO_ROOT / "artifacts" / "drift_reports"


def compute_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    columns: list[str] | None = None,
    label: str = "production",
) -> dict:
    """
    Compute PSI-based drift report between reference and current data.

    Returns a dict with:
      - timestamp
      - label
      - psi_table (DataFrame)
      - summary dict
      - overall_status ('stable' | 'warning' | 'drift')
    """
    psi_df = compute_psi(reference, current, columns=columns)
    summary = psi_summary(psi_df)

    if summary["n_drift"] > 0:
        overall = "drift"
    elif summary["n_warning"] > 0:
        overall = "warning"
    else:
        overall = "stable"

    return {
        "timestamp": datetime.utcnow().isoformat(),
        "label": label,
        "psi_table": psi_df,
        "summary": summary,
        "overall_status": overall,
    }


def save_drift_report(report: dict, output_dir: Path | str | None = None) -> Path:
    """Save drift report to disk (CSV for Power BI + JSON summary)."""
    output_dir = Path(output_dir or DRIFT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    ts = report["timestamp"].replace(":", "-").replace(".", "-")
    csv_path = output_dir / f"drift_{ts}.csv"
    json_path = output_dir / f"drift_{ts}.json"

    psi_df = report["psi_table"].copy()
    psi_df["timestamp"] = report["timestamp"]
    psi_df["overall_status"] = report["overall_status"]
    write_csv(psi_df, csv_path)

    summary_out = {k: v for k, v in report.items() if k != "psi_table"}
    write_json(summary_out, json_path)

    print(f"[drift] Report saved: {csv_path}")
    return csv_path


def generate_powerbi_drift_summary(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    monitoring_window: str = "latest",
) -> pd.DataFrame:
    """
    Generate a Power BI-ready drift summary table.
    Each row = one feature; columns include psi, status, window, timestamp.
    """
    psi_df = compute_psi(reference, current)
    psi_df["monitoring_window"] = monitoring_window
    psi_df["timestamp"] = datetime.utcnow().isoformat()
    psi_df["psi_stable_threshold"] = 0.10
    psi_df["psi_warning_threshold"] = 0.20
    return psi_df
