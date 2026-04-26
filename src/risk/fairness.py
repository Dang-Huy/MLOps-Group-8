"""
src/risk/fairness.py
====================
Fairness diagnostics -- checks for differential performance across subgroups.
Uses Occupation as the available group attribute (PII dropped).
"""
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score

LABEL_MAP = {0: "Poor", 1: "Standard", 2: "Good"}


def fairness_report(
    df: pd.DataFrame,
    y_true_col: str,
    y_pred_col: str,
    group_col: str = "Occupation",
) -> pd.DataFrame:
    """
    Compute per-group classification metrics.

    Parameters
    ----------
    df         : DataFrame with true labels, predictions, and group column
    y_true_col : name of the true label column
    y_pred_col : name of the predicted label column
    group_col  : column to group by (default 'Occupation')

    Returns
    -------
    DataFrame with per-group f1_macro, accuracy, count
    """
    if group_col not in df.columns:
        print(f"[fairness] Group column '{group_col}' not found -- skipping.")
        return pd.DataFrame(columns=["group", "f1_macro", "accuracy", "n_samples"])

    rows = []
    for group, sub in df.groupby(group_col):
        yt = sub[y_true_col].values
        yp = sub[y_pred_col].values
        if len(yt) < 10:
            continue
        rows.append({
            "group":     str(group),
            "n_samples": len(yt),
            "f1_macro":  round(f1_score(yt, yp, average="macro",    zero_division=0), 4),
            "f1_weighted": round(f1_score(yt, yp, average="weighted", zero_division=0), 4),
            "accuracy":  round(accuracy_score(yt, yp), 4),
        })

    result = pd.DataFrame(rows).sort_values("f1_macro") if rows else pd.DataFrame(
        columns=["group", "n_samples", "f1_macro", "f1_weighted", "accuracy"]
    )
    return result.reset_index(drop=True)


def fairness_summary(fairness_df: pd.DataFrame) -> dict:
    """Return a dict highlighting max disparity."""
    if fairness_df.empty:
        return {"status": "skipped", "reason": "group column absent or no subgroup had >=10 samples"}
    return {
        "n_groups":        len(fairness_df),
        "max_f1_macro":    float(fairness_df["f1_macro"].max()),
        "min_f1_macro":    float(fairness_df["f1_macro"].min()),
        "disparity_gap":   round(
            float(fairness_df["f1_macro"].max()) - float(fairness_df["f1_macro"].min()), 4
        ),
        "worst_group":     fairness_df.iloc[0]["group"] if len(fairness_df) else "N/A",
        "best_group":      fairness_df.iloc[-1]["group"] if len(fairness_df) else "N/A",
    }
