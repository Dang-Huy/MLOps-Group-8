"""
src/risk/psi.py
===============
Population Stability Index (PSI) computation.

PSI < 0.10  -> No significant change
PSI 0.10-0.20 -> Moderate change, monitor
PSI > 0.20  -> Significant shift -- investigate
"""
import numpy as np
import pandas as pd


PSI_THRESHOLDS = {"stable": 0.10, "warning": 0.20}


def _psi_for_column(
    expected: np.ndarray,
    actual: np.ndarray,
    bins: int = 10,
    epsilon: float = 1e-6,
) -> float:
    """Compute PSI for a single numeric feature."""
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)

    # Remove NaN
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]

    if len(expected) == 0 or len(actual) == 0:
        return np.nan

    # Build bins from expected (reference) distribution
    breakpoints = np.quantile(expected, np.linspace(0, 1, bins + 1))
    breakpoints = np.unique(breakpoints)  # remove duplicate edges
    if len(breakpoints) < 2:
        return np.nan

    exp_hist, _ = np.histogram(expected, bins=breakpoints)
    act_hist, _ = np.histogram(actual, bins=breakpoints)

    exp_pct = exp_hist / len(expected) + epsilon
    act_pct = act_hist / len(actual) + epsilon

    psi = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
    return float(round(psi, 6))


def _psi_categorical(
    expected: pd.Series,
    actual: pd.Series,
    epsilon: float = 1e-6,
) -> float:
    """Compute PSI for a categorical feature."""
    all_cats = set(expected.dropna().unique()) | set(actual.dropna().unique())
    exp_vc = expected.value_counts(normalize=True)
    act_vc = actual.value_counts(normalize=True)

    psi = 0.0
    for cat in all_cats:
        e = exp_vc.get(cat, 0) + epsilon
        a = act_vc.get(cat, 0) + epsilon
        psi += (a - e) * np.log(a / e)
    return float(round(psi, 6))


def compute_psi(
    reference: pd.DataFrame,
    production: pd.DataFrame,
    columns: list[str] | None = None,
    bins: int = 10,
) -> pd.DataFrame:
    """
    Compute PSI for all specified columns.

    Parameters
    ----------
    reference  : DataFrame from the reference (train) period
    production : DataFrame from production/new period
    columns    : columns to check; defaults to numeric columns present in both
    bins       : histogram bins for numeric PSI

    Returns
    -------
    DataFrame with columns [feature, psi, status]
    """
    if columns is None:
        columns = [
            c for c in reference.columns
            if c in production.columns
            and pd.api.types.is_numeric_dtype(reference[c])
        ]

    rows = []
    for col in columns:
        if col not in reference.columns or col not in production.columns:
            continue
        if pd.api.types.is_numeric_dtype(reference[col]):
            psi = _psi_for_column(reference[col].values, production[col].values, bins=bins)
        else:
            psi = _psi_categorical(reference[col], production[col])

        if np.isnan(psi):
            status = "unknown"
        elif psi < PSI_THRESHOLDS["stable"]:
            status = "stable"
        elif psi < PSI_THRESHOLDS["warning"]:
            status = "warning"
        else:
            status = "drift"

        rows.append({"feature": col, "psi": psi, "status": status})

    return pd.DataFrame(rows, columns=["feature", "psi", "status"])


def psi_summary(psi_df: pd.DataFrame) -> dict:
    """Return a dict summary of PSI results."""
    return {
        "n_features": len(psi_df),
        "n_stable": int((psi_df["status"] == "stable").sum()),
        "n_warning": int((psi_df["status"] == "warning").sum()),
        "n_drift": int((psi_df["status"] == "drift").sum()),
        "max_psi": float(psi_df["psi"].max()),
        "mean_psi": float(psi_df["psi"].mean()),
        "drifted_features": psi_df[psi_df["status"] == "drift"]["feature"].tolist(),
    }
