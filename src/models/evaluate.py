"""
src/models/evaluate.py
======================
Compute evaluation metrics on a fitted model.

Input  : y_true, y_pred, y_prob
Output : metrics dict + JSON report saved to artifacts/reports/
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, confusion_matrix,
    classification_report,
)

REPO_ROOT  = Path(__file__).resolve().parent.parent.parent
REPORT_DIR = REPO_ROOT / "artifacts" / "reports"
LABEL_MAP  = {0: "Poor", 1: "Standard", 2: "Good"}


def _ks_per_class(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    ks = {}
    for k in range(y_prob.shape[1]):
        binary = (y_true == k).astype(int)
        prob_k = y_prob[:, k]
        df = pd.DataFrame({"label": binary, "prob": prob_k}).sort_values("prob", ascending=False)
        n_pos, n_neg = binary.sum(), len(binary) - binary.sum()
        if n_pos == 0 or n_neg == 0:
            ks[LABEL_MAP[k]] = 0.0
            continue
        cum_pos = df["label"].cumsum() / n_pos
        cum_neg = (1 - df["label"]).cumsum() / n_neg
        ks[LABEL_MAP[k]] = round(float((cum_pos - cum_neg).abs().max()), 4)
    return ks


def evaluate(y_true, y_pred, y_prob: np.ndarray, split: str = "valid") -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    report = {
        "split":           split,
        "n_samples":       int(len(y_true)),
        "accuracy":        round(accuracy_score(y_true, y_pred), 4),
        "f1_macro":        round(f1_score(y_true, y_pred, average="macro"), 4),
        "f1_weighted":     round(f1_score(y_true, y_pred, average="weighted"), 4),
        "precision_macro": round(precision_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "recall_macro":    round(recall_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "auc_ovr":         round(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"), 4),
        "auc_ovo":         round(roc_auc_score(y_true, y_prob, multi_class="ovo", average="macro"), 4),
        "ks_per_class":    _ks_per_class(y_true, y_prob),
        "f1_per_class": {
            LABEL_MAP[i]: round(v, 4)
            for i, v in enumerate(f1_score(y_true, y_pred, average=None, zero_division=0))
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "class_labels":     ["Poor", "Standard", "Good"],
    }

    print(f"\n[evaluate] -- {split.upper()} --")
    print(f"  accuracy   : {report['accuracy']}")
    print(f"  f1 (macro) : {report['f1_macro']}")
    print(f"  auc (ovr)  : {report['auc_ovr']}")
    print(f"  ks         : {report['ks_per_class']}")
    print()
    print(classification_report(y_true, y_pred, target_names=["Poor", "Standard", "Good"]))
    return report


def save_report(report: dict, filename: str | None = None) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    filename = filename or f"eval_{report['split']}.json"
    path = REPORT_DIR / filename
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[evaluate] Report saved -> {path}")
    return path