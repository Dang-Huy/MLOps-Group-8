"""
src/models/predict.py
=====================
Run inference on new data using the serialised model bundle.

Input  : raw dict / DataFrame + model bundle path
Output : predicted classes, probabilities, decisions
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.models.serialize import load_bundle, ModelBundle
from src.serving.decision_policy import apply_batch_decisions

REPO_ROOT         = Path(__file__).resolve().parent.parent.parent
DEFAULT_BUNDLE    = REPO_ROOT / "artifacts" / "models" / "final_model_bundle.pkl"
CLASS_LABELS      = ["Poor", "Standard", "Good"]


def predict(
    df: pd.DataFrame,
    bundle: ModelBundle | None = None,
    bundle_path: str | Path | None = None,
    confidence_threshold: float = 0.60,
    return_proba: bool = True,
) -> pd.DataFrame:
    """
    Run end-to-end inference on a DataFrame of raw records.

    Parameters
    ----------
    df                   : DataFrame in raw / serving schema (PII present or absent)
    bundle               : pre-loaded ModelBundle (avoids reloading from disk)
    bundle_path          : path to .pkl bundle (used if bundle is None)
    confidence_threshold : threshold for decision_policy low-confidence override
    return_proba         : include per-class probability columns in output

    Returns
    -------
    DataFrame with columns:
        predicted_class, predicted_label, confidence, action, decision,
        [prob_Poor, prob_Standard, prob_Good]  (if return_proba=True)
    """
    if bundle is None:
        bundle = load_bundle(Path(bundle_path or DEFAULT_BUNDLE))

    X = bundle.transform(df)
    probas = bundle.model.predict_proba(X)           # (N, 3)
    decisions = apply_batch_decisions(probas, confidence_threshold)

    result = pd.DataFrame(decisions)

    if return_proba:
        prob_df = pd.DataFrame(probas, columns=[f"prob_{c}" for c in CLASS_LABELS])
        result = pd.concat([result, prob_df], axis=1)

    return result.reset_index(drop=True)


def predict_from_dict(
    record: dict,
    bundle: ModelBundle | None = None,
    bundle_path: str | Path | None = None,
    confidence_threshold: float = 0.60,
) -> dict:
    """Convenience wrapper for a single dict record."""
    df = pd.DataFrame([record])
    result_df = predict(df, bundle=bundle, bundle_path=bundle_path,
                        confidence_threshold=confidence_threshold)
    return result_df.iloc[0].to_dict()
