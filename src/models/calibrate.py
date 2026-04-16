"""
src/models/calibrate.py
========================
Calibrate probability outputs of a fitted classifier.

Uses sklearn's CalibratedClassifierCV with isotonic regression
(best for large datasets like ours) or Platt scaling (sigmoid).

Input  : fitted model + X_valid, y_valid
Output : calibrated model wrapper + calibration report
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.calibration import CalibratedClassifierCV, calibration_curve

REPO_ROOT  = Path(__file__).resolve().parent.parent.parent
REPORT_DIR = REPO_ROOT / "artifacts" / "reports"
CLASS_LABELS = ["Poor", "Standard", "Good"]


def calibrate_model(
    model,
    X_valid: pd.DataFrame,
    y_valid: np.ndarray,
    method: str = "isotonic",
    cv: int = 5,
) -> CalibratedClassifierCV:
    """
    Wrap a pre-fitted model in CalibratedClassifierCV.

    Parameters
    ----------
    model   : already-fitted classifier (must have a fit method)
    X_valid : calibration / validation features
    y_valid : true labels
    method  : 'isotonic' or 'sigmoid' (Platt scaling)
    cv      : number of cross-validation folds (default 5)
              The model's fit() is called per fold; for ensemble wrappers
              with a no-op fit(), this effectively calibrates on held-out
              out-of-fold predictions from the original trained model.

    Returns
    -------
    CalibratedClassifierCV fitted on (X_valid, y_valid)
    """
    cal = CalibratedClassifierCV(estimator=model, method=method, cv=cv)
    cal.fit(X_valid, y_valid)
    print(f"[calibrate] Calibrated with method='{method}', cv={cv} on {len(y_valid)} samples.")
    return cal


def calibration_report(
    y_true: np.ndarray,
    y_prob_before: np.ndarray,
    y_prob_after:  np.ndarray,
    n_bins: int = 10,
) -> dict:
    """
    Compute Expected Calibration Error (ECE) before and after calibration.

    Parameters
    ----------
    y_true        : integer true labels (0, 1, 2)
    y_prob_before : (N, 3) probability matrix before calibration
    y_prob_after  : (N, 3) probability matrix after calibration

    Returns
    -------
    dict with ECE values per class and overall, before/after
    """
    def _ece(y_bin, prob, n_bins):
        """Expected Calibration Error for one binary sub-problem."""
        bins = np.linspace(0, 1, n_bins + 1)
        ece  = 0.0
        n    = len(y_bin)
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (prob >= lo) & (prob < hi)
            if mask.sum() == 0:
                continue
            avg_conf = prob[mask].mean()
            avg_acc  = y_bin[mask].mean()
            ece += mask.sum() / n * abs(avg_conf - avg_acc)
        return round(float(ece), 4)

    report = {"n_samples": len(y_true), "n_bins": n_bins}
    for k, cls in enumerate(CLASS_LABELS):
        y_bin = (y_true == k).astype(int)
        report[f"ece_before_{cls}"] = _ece(y_bin, y_prob_before[:, k], n_bins)
        report[f"ece_after_{cls}"]  = _ece(y_bin, y_prob_after[:, k],  n_bins)

    before_overall = np.mean([report[f"ece_before_{c}"] for c in CLASS_LABELS])
    after_overall  = np.mean([report[f"ece_after_{c}"]  for c in CLASS_LABELS])
    report["ece_before_mean"] = round(float(before_overall), 4)
    report["ece_after_mean"]  = round(float(after_overall),  4)
    print(f"[calibrate] ECE before={before_overall:.4f}  after={after_overall:.4f}")
    return report


def plot_calibration_curve(
    y_true: np.ndarray,
    y_prob_before: np.ndarray,
    y_prob_after:  np.ndarray,
    n_bins: int = 10,
    save_path: str | Path | None = None,
) -> Path:
    """Plot reliability diagrams for all 3 classes, before/after."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    for k, (cls, ax) in enumerate(zip(CLASS_LABELS, axes)):
        y_bin = (y_true == k).astype(int)
        for proba, label, color, ls in [
            (y_prob_before[:, k], "Before calibration", "#EF5350", "--"),
            (y_prob_after[:, k],  "After calibration",  "#42A5F5", "-"),
        ]:
            frac_pos, mean_pred = calibration_curve(y_bin, proba, n_bins=n_bins)
            ax.plot(mean_pred, frac_pos, marker="o", label=label, color=color, linestyle=ls)

        ax.plot([0, 1], [0, 1], "k--", label="Perfect", linewidth=0.8)
        ax.set_title(f"Class: {cls}", fontsize=12)
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction of positives")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.suptitle("Calibration Reliability Diagrams", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path is None:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        save_path = REPORT_DIR / "calibration_curve.png"
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[calibrate] Calibration curve saved -> {save_path}")
    return save_path
