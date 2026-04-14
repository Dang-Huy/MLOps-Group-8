"""
src/models/train.py
===================
Fit the classification model on the prepared feature matrix.

Input  : X_train, y_train + configs/train.yaml
Output : fitted model object + training metadata dict
Role   : Single entry point for model fitting; no data loading here.
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_sample_weight

REPO_ROOT    = Path(__file__).resolve().parent.parent.parent
ARTIFACT_DIR = REPO_ROOT / "artifacts" / "models"

DEFAULT_PARAMS = {
    "model_type":       "xgboost",
    "max_depth":        4,
    "learning_rate":    0.05,
    "n_estimators":     300,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "random_state":     42,
    "n_jobs":           -1,
    "verbosity":        0,
    "num_class":        3,
    "objective":        "multi:softprob",
    "eval_metric":      "mlogloss",
    "early_stopping_rounds": 20,   # constructor param in xgboost >= 1.6
}


def train(
    X_train,
    y_train,
    X_valid=None,
    y_valid=None,
    params: dict | None = None,
) -> tuple[XGBClassifier, dict]:
    """
    Fit an XGBoost multiclass classifier with balanced class weights.
    EDA found: Good 17.8%, Standard 53.2%, Poor 29.0%.

    early_stopping_rounds is passed to the constructor (xgboost >= 1.6).
    If no validation set is provided, early stopping is disabled.
    """
    cfg = {**DEFAULT_PARAMS, **(params or {})}
    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)

    use_early_stopping = (X_valid is not None and y_valid is not None)

    model = XGBClassifier(
        max_depth             = cfg["max_depth"],
        learning_rate         = cfg["learning_rate"],
        n_estimators          = cfg["n_estimators"],
        subsample             = cfg["subsample"],
        colsample_bytree      = cfg["colsample_bytree"],
        random_state          = cfg["random_state"],
        n_jobs                = cfg["n_jobs"],
        verbosity             = cfg["verbosity"],
        num_class             = cfg["num_class"],
        objective             = cfg["objective"],
        eval_metric           = cfg["eval_metric"],
        early_stopping_rounds = cfg["early_stopping_rounds"] if use_early_stopping else None,
    )

    fit_kwargs: dict = {"sample_weight": sample_weights}
    if use_early_stopping:
        fit_kwargs["eval_set"] = [(X_valid, y_valid)]
        fit_kwargs["verbose"]  = False

    t0 = time.time()
    model.fit(X_train, y_train, **fit_kwargs)
    elapsed = round(time.time() - t0, 2)

    n_used = (model.best_iteration + 1
              if use_early_stopping and hasattr(model, "best_iteration") and model.best_iteration
              else cfg["n_estimators"])

    metadata = {
        "model_type":        cfg["model_type"],
        "params":            cfg,
        "n_features":        X_train.shape[1],
        "n_train_rows":      int(len(y_train)),
        "n_estimators_used": n_used,
        "training_time_sec": elapsed,
        "class_weighting":   "balanced",
    }

    print(f"[train] Fitted in {elapsed}s — {n_used} trees, {X_train.shape[1]} features")
    return model, metadata


def save_metadata(metadata: dict, path: Path | None = None) -> None:
    path = path or (ARTIFACT_DIR / "train_metadata.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[train] Metadata saved → {path}")