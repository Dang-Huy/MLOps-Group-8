"""
src/risk/stability.py
=====================
Score and performance stability analysis.
Measures how consistently the model scores / performs over time.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score


def score_stability(
    scores: list[np.ndarray],
    labels: list[str] | None = None,
    bins: int = 10,
) -> pd.DataFrame:
    """
    Compare prediction probability distributions across periods.

    Parameters
    ----------
    scores : list of 1-D arrays (one per period)
    labels : optional period labels
    bins   : histogram bins

    Returns
    -------
    DataFrame with statistics per period
    """
    labels = labels or [f"period_{i}" for i in range(len(scores))]
    rows = []
    for lbl, s in zip(labels, scores):
        s = np.asarray(s)
        rows.append({
            "period": lbl,
            "mean_score": float(np.nanmean(s)),
            "std_score": float(np.nanstd(s)),
            "p10": float(np.nanpercentile(s, 10)),
            "p50": float(np.nanpercentile(s, 50)),
            "p90": float(np.nanpercentile(s, 90)),
        })
    return pd.DataFrame(rows)


def performance_stability(
    y_trues: list[np.ndarray],
    y_preds: list[np.ndarray],
    labels: list[str] | None = None,
) -> pd.DataFrame:
    """Compare classification metrics across periods."""
    labels = labels or [f"period_{i}" for i in range(len(y_trues))]
    rows = []
    for lbl, yt, yp in zip(labels, y_trues, y_preds):
        rows.append({
            "period": lbl,
            "f1_macro": round(f1_score(yt, yp, average="macro", zero_division=0), 4),
            "f1_weighted": round(f1_score(yt, yp, average="weighted", zero_division=0), 4),
            "accuracy": round(accuracy_score(yt, yp), 4),
        })
    return pd.DataFrame(rows)
