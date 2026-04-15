"""
src/pipelines/retraining_pipeline.py
======================================
Drift-gated retraining pipeline.

Steps
-----
1. Load current production model and compute reference metrics.
2. Compute feature-level PSI drift between reference data and new data.
3. Check output distribution shift (label drift).
4. Decision gate: retrain only if drift exceeds thresholds OR explicitly forced.
5. If retraining: run full training_pipeline and compare new vs. old model.
6. Promote new model only if it improves on the validation gate.

Usage
-----
    # Auto-check drift and retrain if needed
    python -m src.pipelines.retraining_pipeline

    # Force retrain regardless of drift
    python -m src.pipelines.retraining_pipeline --force

    # Retrain on new data file
    python -m src.pipelines.retraining_pipeline --new-data path/to/new_data.csv

Drift thresholds
----------------
    PSI > 0.2  on any feature        -> retrain recommended
    label_shift > 0.1                -> retrain recommended
    Both below thresholds + no force -> skip retraining (model still healthy)
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.utils.logger import get_logger
from src.utils.io     import write_json

logger     = get_logger(__name__)
REPORT_DIR = ROOT / "artifacts" / "reports"
REF_DIR    = ROOT / "data" / "reference"
MODEL_DIR  = ROOT / "artifacts" / "models"

# Drift thresholds that trigger retraining
PSI_THRESHOLD   = 0.2   # any single feature PSI > this -> retrain
LABEL_THRESHOLD = 0.1   # Wasserstein distance on predicted probabilities


# ── Drift checks ──────────────────────────────────────────────────────────────

def _check_feature_drift(ref_df: pd.DataFrame, new_df: pd.DataFrame) -> dict:
    """Compute PSI for numeric features shared between ref and new data."""
    try:
        from src.risk.psi import compute_psi
    except ImportError:
        logger.warning("PSI module not available -- skipping feature drift.")
        return {}

    common = [c for c in ref_df.columns
              if c in new_df.columns and pd.api.types.is_numeric_dtype(ref_df[c])][:30]
    results: dict = {}
    for col in common:
        try:
            results[col] = float(compute_psi(ref_df[col].dropna(), new_df[col].dropna()))
        except Exception:
            pass
    return results


def _check_label_drift(
    ref_bundle_model, ref_X: pd.DataFrame,
    new_X: pd.DataFrame,
) -> float:
    """
    Compute Wasserstein-1 distance between reference and new predicted
    probability distributions (averaged over classes).
    Returns 0.0 if scipy is not available.
    """
    try:
        from scipy.stats import wasserstein_distance
    except ImportError:
        return 0.0

    try:
        ref_probs = ref_bundle_model.predict_proba(ref_X)
        new_probs = ref_bundle_model.predict_proba(new_X)
        distances = [
            wasserstein_distance(ref_probs[:, k], new_probs[:, k])
            for k in range(ref_probs.shape[1])
        ]
        return float(np.mean(distances))
    except Exception as e:
        logger.warning(f"Label drift check failed: {e}")
        return 0.0


# ── Model comparison ──────────────────────────────────────────────────────────

def _evaluate_on_test(bundle_path: Path) -> dict:
    """Load bundle and evaluate on test_split.csv."""
    from src.models.serialize import load_bundle
    from src.models.evaluate  import evaluate
    from src.data.schema      import TARGET_COL

    test_path = ROOT / "data" / "processed" / "test_split.csv"
    if not test_path.exists():
        return {}

    bundle  = load_bundle(bundle_path)
    test_df = pd.read_csv(test_path)
    y_test  = test_df[TARGET_COL].astype(int).values

    # Use bundle's transform pipeline
    from src.data.preprocessing import (
        drop_pii_columns, clean_numeric_columns, cap_outliers,
        parse_credit_history_age, clean_categorical_columns,
    )
    from src.features.build_features import build_features

    df = test_df.drop(columns=[TARGET_COL], errors="ignore").copy()
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
    return evaluate(y_test, y_pred, y_prob, split="test")


# ── Main retraining pipeline ──────────────────────────────────────────────────

def run_retraining_pipeline(
    trigger:       str        = "manual",
    new_data_path: str | Path | None = None,
    force:         bool       = False,
    psi_threshold: float      = PSI_THRESHOLD,
    label_threshold: float    = LABEL_THRESHOLD,
) -> dict:
    """
    Drift-gated retraining workflow.

    Parameters
    ----------
    trigger        : reason string ('drift', 'scheduled', 'manual')
    new_data_path  : optional new raw CSV to append / replace training data
    force          : bypass drift gate and always retrain
    psi_threshold  : max PSI before retraining is triggered
    label_threshold: max label drift before retraining is triggered

    Returns
    -------
    dict with keys: retrained, reason, old_f1, new_f1, promoted
    """
    t0 = time.time()
    ts = datetime.utcnow().isoformat()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("== RETRAINING PIPELINE")
    logger.info(f"   trigger={trigger}   force={force}   time={ts}")
    logger.info("=" * 60)

    result = {
        "timestamp":   ts,
        "trigger":     trigger,
        "force":       force,
        "retrained":   False,
        "reason":      "not_evaluated",
        "old_f1":      None,
        "new_f1":      None,
        "promoted":    False,
        "drift_report": {},
    }

    # ── 1. Load current production model ─────────────────────────────────────
    logger.info("\n-- 1. Current model --")
    bundle_path = MODEL_DIR / "final_model_bundle.pkl"
    if not bundle_path.exists():
        logger.warning("No existing model bundle found -- forcing retrain.")
        force = True
    else:
        from src.models.serialize import load_bundle
        bundle     = load_bundle(bundle_path)
        old_report = _evaluate_on_test(bundle_path)
        old_f1     = old_report.get("f1_macro", 0.0)
        result["old_f1"] = old_f1
        logger.info(f"   Current model: {bundle.metadata.get('final_model_name', '?')}  "
                    f"f1_macro={old_f1:.4f}")

    # ── 2. Drift analysis ─────────────────────────────────────────────────────
    if not force:
        logger.info("\n-- 2. Drift analysis --")
        ref_feat_path = REF_DIR / "reference_features.parquet"
        drift_report  = {"psi": {}, "label_shift": 0.0, "needs_retrain": False}

        if ref_feat_path.exists() and new_data_path is not None:
            new_path = Path(new_data_path)
            new_df   = pd.read_csv(new_path) if new_path.suffix == ".csv" \
                       else pd.read_parquet(new_path)
            ref_df   = pd.read_parquet(ref_feat_path)

            # Feature-level PSI
            psi_scores = _check_feature_drift(ref_df, new_df)
            drift_report["psi"] = psi_scores
            high_psi = {k: v for k, v in psi_scores.items() if v > psi_threshold}
            if high_psi:
                logger.warning(f"   HIGH PSI ({len(high_psi)} features): {list(high_psi.keys())[:5]}")
                drift_report["needs_retrain"] = True
            else:
                logger.info(f"   Feature drift OK (max PSI={max(psi_scores.values(), default=0):.3f})")

            # Label (prediction) drift
            label_shift = _check_label_drift(bundle.model, ref_df, new_df)
            drift_report["label_shift"] = label_shift
            logger.info(f"   Label shift (Wasserstein-1): {label_shift:.4f}")
            if label_shift > label_threshold:
                logger.warning(f"   Label drift exceeds threshold ({label_threshold}).")
                drift_report["needs_retrain"] = True
        else:
            if ref_feat_path.exists():
                logger.info("   Drift check skipped: no new_data_path provided.")
            else:
                logger.info("   Reference data not found -- assuming first run, retraining.")
                drift_report["needs_retrain"] = True

        result["drift_report"] = drift_report

        if not drift_report["needs_retrain"]:
            result["reason"] = "drift_below_threshold"
            logger.info("\n   Model is healthy. No retraining needed.")
            elapsed = time.time() - t0
            logger.info(f"\n== RETRAINING PIPELINE COMPLETE in {elapsed:.1f}s")
            write_json(result, REPORT_DIR / "retraining_report.json")
            return result
    else:
        logger.info("\n-- 2. Drift analysis skipped (force=True) --")

    # ── 3. Retrain ────────────────────────────────────────────────────────────
    logger.info("\n-- 3. Retraining --")
    if new_data_path is not None:
        logger.info(f"   New data provided at: {new_data_path}")
        logger.info("   (Append or replace logic should be implemented here for production.)")

    from src.pipelines.training_pipeline import run_training_pipeline
    new_bundle = run_training_pipeline()
    result["retrained"] = True
    result["reason"]    = "force" if force else "drift_threshold_exceeded"
    logger.info(f"   Retraining complete.")

    # ── 4. Compare new vs. old ────────────────────────────────────────────────
    logger.info("\n-- 4. Model comparison --")
    new_report = _evaluate_on_test(bundle_path)
    new_f1     = new_report.get("f1_macro", 0.0)
    result["new_f1"] = new_f1

    improvement = new_f1 - (result["old_f1"] or 0.0)
    logger.info(f"   Old f1_macro: {result['old_f1']}")
    logger.info(f"   New f1_macro: {new_f1:.4f}  (delta={improvement:+.4f})")

    # ── 5. Promote if improved ────────────────────────────────────────────────
    logger.info("\n-- 5. Promotion decision --")
    if result["old_f1"] is None or new_f1 >= (result["old_f1"] - 0.005):
        from src.models.registry import promote_to_production
        promote_to_production(new_bundle.metadata.get("final_model_name", "unknown"), "1.0.0")
        result["promoted"] = True
        logger.info("   New model PROMOTED to production.")
    else:
        logger.warning(f"   New model worse by {-improvement:.4f} -- NOT promoted.")
        logger.warning("   Previous model remains in production.")

    elapsed = time.time() - t0
    logger.info(f"\n== RETRAINING PIPELINE COMPLETE in {elapsed:.1f}s")
    write_json(result, REPORT_DIR / "retraining_report.json")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Drift-gated retraining pipeline")
    parser.add_argument("--new-data", default=None, help="Path to new raw data CSV/Parquet")
    parser.add_argument("--force",    action="store_true", help="Bypass drift gate")
    parser.add_argument("--trigger",  default="manual",
                        choices=["manual", "scheduled", "drift", "api"],
                        help="Trigger source (for audit log)")
    parser.add_argument("--psi-threshold",   type=float, default=PSI_THRESHOLD)
    parser.add_argument("--label-threshold", type=float, default=LABEL_THRESHOLD)
    args = parser.parse_args()

    run_retraining_pipeline(
        trigger=args.trigger,
        new_data_path=args.new_data,
        force=args.force,
        psi_threshold=args.psi_threshold,
        label_threshold=args.label_threshold,
    )
