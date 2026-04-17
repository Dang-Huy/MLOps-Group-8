"""
src/monitoring/drift_monitor.py
================================
Production-vs-reference drift checks using PSI.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.risk.psi import compute_psi, psi_summary
from src.utils.io import write_csv, read_parquet, write_json
from src.utils.logger import get_logger

logger = get_logger(__name__)

REPO_ROOT      = Path(__file__).resolve().parent.parent.parent
REFERENCE_PATH = REPO_ROOT / "data" / "reference" / "train_reference.parquet"
DRIFT_DIR      = REPO_ROOT / "artifacts" / "drift_reports"


def run_drift_check(
    current_df:     pd.DataFrame,
    reference_path: str | Path | None = None,
    output_dir:     str | Path | None = None,
    label:          str = "production",
) -> dict:
    """
    Compare current_df against the reference distribution using PSI.
    Saves a Power BI-ready CSV to artifacts/drift_reports/.

    Returns a report dict with psi_table, summary, overall_status.
    """
    ref_path = Path(reference_path or REFERENCE_PATH)
    if not ref_path.exists():
        logger.warning(
            f"[drift_monitor] Reference not found at {ref_path} -- skipping drift check."
        )
        return {"status": "skipped", "reason": "reference_missing"}

    reference = read_parquet(ref_path)

    # Intersect numeric columns present in both
    num_cols = [
        c for c in reference.columns
        if c in current_df.columns and pd.api.types.is_numeric_dtype(reference[c])
    ]

    psi_df  = compute_psi(reference, current_df, columns=num_cols)
    summary = psi_summary(psi_df)

    if summary["n_drift"] > 0:
        overall = "drift"
    elif summary["n_warning"] > 0:
        overall = "warning"
    else:
        overall = "stable"

    from datetime import datetime
    ts = datetime.utcnow().isoformat()

    out_dir = Path(output_dir or DRIFT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Power BI export
    pbi_df = psi_df.copy()
    pbi_df["monitoring_window"]  = label
    pbi_df["timestamp"]          = ts
    pbi_df["overall_status"]     = overall
    pbi_df["psi_stable_thresh"]  = 0.10
    pbi_df["psi_warning_thresh"] = 0.20

    pbi_path = out_dir / "powerbi_drift_summary.csv"
    write_csv(pbi_df, pbi_path)
    logger.info(f"[drift_monitor] Drift summary saved -> {pbi_path}")

    report = {
        "timestamp":      ts,
        "label":          label,
        "psi_table":      psi_df,
        "summary":        summary,
        "overall_status": overall,
    }
    return report
