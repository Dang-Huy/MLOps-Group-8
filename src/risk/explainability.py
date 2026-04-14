"""
src/risk/explainability.py
==========================
SHAP-based feature explainability for the credit score classifier.
Falls back to built-in feature_importances_ if SHAP is unavailable.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT  = Path(__file__).resolve().parent.parent.parent
REPORT_DIR = REPO_ROOT / "artifacts" / "reports"


def global_feature_importance(
    model,
    X: pd.DataFrame,
    method: str = "shap",
    background_samples: int = 200,
) -> pd.DataFrame:
    """
    Compute global feature importance.

    Parameters
    ----------
    model            : fitted sklearn-compatible model
    X                : feature DataFrame (used for SHAP background)
    method           : 'shap' | 'builtin' -- falls back to builtin on error
    background_samples : number of background rows for SHAP TreeExplainer

    Returns
    -------
    DataFrame [feature, importance] sorted descending
    """
    if method == "shap":
        try:
            import shap
            bg = X.sample(min(background_samples, len(X)), random_state=42)
            explainer = shap.TreeExplainer(model, bg)
            shap_vals = explainer.shap_values(X)

            # Handle multiple return types across shap versions:
            # - shap.Explanation object (shap >= 0.45 new style)
            # - list of 2D arrays  (old multiclass style: one array per class)
            # - 3D numpy array     (n_samples, n_features, n_classes)
            # - 2D numpy array     (binary / regression)
            if hasattr(shap_vals, "values"):
                # shap.Explanation object
                vals = np.array(shap_vals.values)
                if vals.ndim == 3:
                    mean_abs = np.abs(vals).mean(axis=(0, 2))
                else:
                    mean_abs = np.abs(vals).mean(axis=0)
            elif isinstance(shap_vals, list):
                mean_abs = np.mean(
                    [np.abs(sv).mean(axis=0) for sv in shap_vals], axis=0
                )
            else:
                arr = np.array(shap_vals)
                if arr.ndim == 3:
                    mean_abs = np.abs(arr).mean(axis=(0, 2))
                else:
                    mean_abs = np.abs(arr).mean(axis=0)

            imp_df = (
                pd.DataFrame({"feature": list(X.columns), "mean_abs_shap": mean_abs})
                .sort_values("mean_abs_shap", ascending=False)
                .reset_index(drop=True)
            )
            imp_df["importance"] = imp_df["mean_abs_shap"]
            return imp_df
        except Exception as e:
            print(f"[explainability] SHAP failed ({e}), falling back to built-in importance.")
            method = "builtin"

    if method == "builtin":
        if hasattr(model, "feature_importances_"):
            imp = model.feature_importances_
        elif hasattr(model, "coef_"):
            imp = np.abs(model.coef_).mean(axis=0)
        else:
            raise ValueError("Model has no built-in feature_importances_ or coef_.")
        return (
            pd.DataFrame({"feature": list(X.columns), "importance": imp})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    raise ValueError(f"Unknown method: {method}")


def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 20,
    save_path: str | Path | None = None,
    title: str = "Global Feature Importance",
) -> Path:
    """Plot horizontal bar chart of feature importance."""
    df = importance_df.head(top_n).copy().sort_values("importance", ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(6, len(df) * 0.35)))
    bars = ax.barh(df["feature"], df["importance"], color="#1976D2", edgecolor="white")
    ax.set_xlabel("Mean |SHAP| / Importance", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    for bar in bars:
        w = bar.get_width()
        ax.text(w + w * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{w:.4f}", va="center", fontsize=8)
    plt.tight_layout()

    if save_path is None:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        save_path = REPORT_DIR / "feature_importance.png"
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[explainability] Feature importance saved -> {save_path}")
    return save_path


def plot_shap_summary(
    model,
    X: pd.DataFrame,
    save_path: str | Path | None = None,
    background_samples: int = 100,
    max_display: int = 20,
) -> Path | None:
    """Generate SHAP beeswarm/dot plot (requires shap package)."""
    try:
        import shap
        bg = X.sample(min(background_samples, len(X)), random_state=42)
        explainer = shap.TreeExplainer(model, bg)
        shap_vals = explainer.shap_values(X)

        if save_path is None:
            REPORT_DIR.mkdir(parents=True, exist_ok=True)
            save_path = REPORT_DIR / "shap_summary.png"
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(10, 8))
        if isinstance(shap_vals, list):
            shap.summary_plot(
                shap_vals, X, plot_type="bar",
                class_names=["Poor", "Standard", "Good"],
                max_display=max_display, show=False,
            )
        else:
            shap.summary_plot(shap_vals, X, max_display=max_display, show=False)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[explainability] SHAP summary saved -> {save_path}")
        return save_path
    except Exception as e:
        print(f"[explainability] SHAP summary plot failed: {e}")
        return None
