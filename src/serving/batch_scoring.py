"""
src/serving/batch_scoring.py
=============================
Score a CSV/parquet file of records and save predictions.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.models.serialize import load_bundle
from src.serving.decision_policy import apply_batch_decisions
from src.utils.io import write_csv
from src.utils.logger import get_logger

logger = get_logger(__name__)

REPO_ROOT        = Path(__file__).resolve().parent.parent.parent
PREDICTIONS_DIR  = REPO_ROOT / "artifacts" / "predictions"
DEFAULT_BUNDLE   = REPO_ROOT / "artifacts" / "models" / "final_model_bundle.pkl"


def score_file(
    input_path:           str | Path,
    model_path:           str | Path | None = None,
    output_path:          str | Path | None = None,
    confidence_threshold: float = 0.60,
) -> pd.DataFrame:
    """
    Score a CSV or parquet file and save predictions alongside original features.

    Parameters
    ----------
    input_path           : path to input file (.csv or .parquet)
    model_path           : path to model bundle pickle
    output_path          : where to save predictions CSV
    confidence_threshold : decision policy confidence threshold

    Returns
    -------
    DataFrame with original data + pred_* columns
    """
    input_path = Path(input_path)
    model_path = Path(model_path or DEFAULT_BUNDLE)

    if input_path.suffix == ".parquet":
        df = pd.read_parquet(input_path)
    else:
        df = pd.read_csv(input_path, low_memory=False)

    logger.info(f"[batch_scoring] Loaded {len(df)} rows from {input_path}")

    bundle = load_bundle(model_path)
    X      = bundle.transform(df)
    probas = bundle.model.predict_proba(X)

    decisions  = apply_batch_decisions(probas, confidence_threshold)
    results_df = pd.DataFrame(decisions)

    out = df.reset_index(drop=True).copy()
    for col in results_df.columns:
        out[f"pred_{col}"] = results_df[col].values

    if output_path is None:
        PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
        output_path = PREDICTIONS_DIR / f"batch_predictions_{input_path.stem}.csv"

    write_csv(out, output_path)
    logger.info(f"[batch_scoring] Saved {len(out)} predictions -> {output_path}")
    return out


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--model",  default=None)
    args = parser.parse_args()
    score_file(args.input, model_path=args.model, output_path=args.output)
