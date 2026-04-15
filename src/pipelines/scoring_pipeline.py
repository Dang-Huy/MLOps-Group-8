"""
src/pipelines/scoring_pipeline.py
===================================
Batch scoring pipeline.

Steps
-----
1. Load input data (raw or pre-processed CSV / Parquet)
2. Run input quality check (missing rate, type drift)
3. Score with the production model bundle
4. Run output distribution check (prediction drift vs. reference)
5. Save predictions + monitoring reports

Usage
-----
    # Score the held-out test split (default)
    python -m src.pipelines.scoring_pipeline

    # Score a new file
    python -m src.pipelines.scoring_pipeline --input path/to/new_data.csv

    # Score against a custom model bundle
    python -m src.pipelines.scoring_pipeline --model path/to/bundle.pkl
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.serving.batch_scoring    import score_file
from src.monitoring.input_monitor import monitor_input, save_input_monitor_report
from src.monitoring.output_monitor import monitor_output
from src.monitoring.drift_monitor  import run_drift_check
from src.utils.logger              import get_logger
from src.utils.io                  import write_json, write_csv

logger     = get_logger(__name__)
REPORT_DIR = ROOT / "artifacts" / "reports"
PRED_DIR   = ROOT / "artifacts" / "predictions"
REF_DIR    = ROOT / "data" / "reference"


def run_scoring_pipeline(
    input_path:  str | Path | None = None,
    output_path: str | Path | None = None,
    model_path:  str | Path | None = None,
    run_drift:   bool = True,
) -> dict:
    """
    Score a dataset and produce monitoring reports.

    Parameters
    ----------
    input_path  : raw CSV to score; defaults to data/processed/test_split.csv
    output_path : where to write predictions CSV
    model_path  : path to model bundle; defaults to production bundle
    run_drift   : whether to compute PSI drift vs. reference data

    Returns
    -------
    dict with keys: output_path, n_rows, class_distribution, drift_summary
    """
    t0 = time.time()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)

    input_path  = Path(input_path  or ROOT / "data" / "processed" / "test_split.csv")
    output_path = Path(output_path or PRED_DIR / "batch_predictions.csv")
    bundle_path = Path(model_path  or ROOT / "artifacts" / "models" / "final_model_bundle.pkl")

    logger.info("=" * 60)
    logger.info("== SCORING PIPELINE")
    logger.info(f"   Input : {input_path}")
    logger.info(f"   Model : {bundle_path}")
    logger.info("=" * 60)

    # ── 1. Load input ─────────────────────────────────────────────────────────
    logger.info("\n-- 1. Load input --")
    if input_path.suffix == ".parquet":
        df_in = pd.read_parquet(input_path)
    else:
        df_in = pd.read_csv(input_path)
    logger.info(f"   Loaded {len(df_in):,} rows x {df_in.shape[1]} cols")

    # ── 2. Input quality check ────────────────────────────────────────────────
    logger.info("\n-- 2. Input quality check --")
    try:
        input_report = monitor_input(df_in)
        save_input_monitor_report(input_report, REPORT_DIR / "scoring_input_quality.csv")
        n_high_miss = sum(1 for r in input_report if r.get("missing_rate", 0) > 0.3)
        if n_high_miss:
            logger.warning(f"   {n_high_miss} feature(s) have >30% missing values.")
        else:
            logger.info("   Input quality OK.")
    except Exception as e:
        logger.warning(f"   Input quality check failed: {e}")

    # ── 3. Score ──────────────────────────────────────────────────────────────
    logger.info("\n-- 3. Scoring --")
    score_file(input_path, model_path=bundle_path, output_path=output_path)
    logger.info(f"   Predictions saved -> {output_path}")

    # Load predictions back for monitoring
    preds_df = pd.read_csv(output_path)
    n_rows = len(preds_df)

    # Class distribution of predictions
    class_dist: dict = {}
    if "predicted_label" in preds_df.columns:
        vc = preds_df["predicted_label"].value_counts(normalize=True).round(3)
        class_dist = vc.to_dict()
        logger.info(f"   Prediction distribution: {class_dist}")

    # ── 4. Output drift check ─────────────────────────────────────────────────
    drift_summary: dict = {}
    if run_drift:
        logger.info("\n-- 4. Output drift check (vs. reference) --")

        # Output distribution (prediction stats for this batch)
        if "predicted_label" in preds_df.columns:
            try:
                pred_labels = preds_df["predicted_label"].values
                prob_cols   = [c for c in preds_df.columns if c.startswith("prob_")]
                probabilities = preds_df[prob_cols].values if prob_cols else None
                out_report = monitor_output(pred_labels, probabilities, label="scoring_batch")
                drift_summary = out_report
                write_json(out_report, REPORT_DIR / "scoring_output_distribution.json")
                logger.info(f"   Predicted distribution: {out_report.get('percentages', {})}")
            except Exception as e:
                logger.warning(f"   Output distribution check failed: {e}")

        # Feature-level PSI drift vs. reference
        try:
            psi_report = run_drift_check(df_in, label="scoring_batch")
            status = psi_report.get("overall_status", "unknown")
            write_json(psi_report, REPORT_DIR / "scoring_feature_drift.json")
            if status == "ALERT":
                logger.warning(f"   Feature drift ALERT -- PSI exceeds threshold.")
            elif status == "skipped":
                logger.info("   Feature drift check skipped (no reference found).")
            else:
                logger.info(f"   Feature drift status: {status}")
            drift_summary["feature_drift_status"] = status
        except Exception as e:
            logger.warning(f"   Feature drift check failed: {e}")

    # ── 5. Summary report ─────────────────────────────────────────────────────
    elapsed = time.time() - t0
    summary = {
        "input_path":        str(input_path),
        "output_path":       str(output_path),
        "n_rows":            n_rows,
        "class_distribution": class_dist,
        "drift_summary":     drift_summary,
        "elapsed_s":         round(elapsed, 2),
    }
    write_json(summary, REPORT_DIR / "scoring_summary.json")
    logger.info(f"\n== SCORING COMPLETE in {elapsed:.1f}s  ({n_rows:,} rows)")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch scoring pipeline")
    parser.add_argument("--input",   default=None, help="Input CSV/Parquet path")
    parser.add_argument("--output",  default=None, help="Output predictions CSV path")
    parser.add_argument("--model",   default=None, help="Model bundle .pkl path")
    parser.add_argument("--no-drift", action="store_true", help="Skip drift check")
    args = parser.parse_args()

    run_scoring_pipeline(
        input_path=args.input,
        output_path=args.output,
        model_path=args.model,
        run_drift=not args.no_drift,
    )
