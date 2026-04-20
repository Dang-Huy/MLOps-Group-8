"""
src/pipelines/hyperparameter_tuning_pipeline.py
================================================
Standalone hyperparameter optimisation (HPO) pipeline.

Runs thorough Optuna studies for each candidate model and persists the
best parameters so that training_pipeline.py can load them directly
instead of re-running Optuna.

Architecture
------------
* Each model gets its own Optuna study stored in a SQLite DB so studies
  can be paused and resumed across runs.
* 3-fold cross-validation on the training set is used to score each
  trial -- this gives more robust estimates than a single validation fold.
* MedianPruner kills unpromising trials early to save time.
* Best params are written to:
    artifacts/reports/best_hyperparameters.json   <- loaded by training_pipeline
    artifacts/reports/hpo_study_results.json      <- full trial history
* Convergence plots are saved to artifacts/reports/hpo_<model>.png

Usage
-----
    # Tune all models (default: 50 trials, 300 s timeout per model)
    python -m src.pipelines.hyperparameter_tuning_pipeline

    # Tune one model with custom budget
    python -m src.pipelines.hyperparameter_tuning_pipeline \\
        --model lightgbm --n-trials 100 --timeout 600

    # Fresh start (delete existing studies)
    python -m src.pipelines.hyperparameter_tuning_pipeline --fresh
"""
from __future__ import annotations

# ruff: noqa: E402

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils.logger import get_logger

logger = get_logger(__name__)

REPORT_DIR = ROOT / "artifacts" / "reports"
STUDY_DIR  = ROOT / "artifacts" / "optuna"
BEST_PARAMS_PATH = REPORT_DIR / "best_hyperparameters.json"
FULL_RESULTS_PATH = REPORT_DIR / "hpo_study_results.json"
SEED = 42

# ── Model names available for tuning ──────────────────────────────────────────
ALL_MODELS = ["lightgbm", "xgboost", "random_forest", "extra_trees", "logistic_regression"]


# ── Search spaces ─────────────────────────────────────────────────────────────

def _suggest_params(trial, name: str) -> dict:
    """Return suggested hyperparameter dict for the given model name."""
    if name == "logistic_regression":
        return {
            "C":           trial.suggest_float("C", 1e-4, 100.0, log=True),
            "solver":      trial.suggest_categorical("solver", ["lbfgs", "saga"]),
            "max_iter":    trial.suggest_int("max_iter", 500, 3000),
        }

    if name == "random_forest":
        return {
            "n_estimators":     trial.suggest_int("n_estimators", 100, 800),
            "max_depth":        trial.suggest_int("max_depth", 5, 30),
            "min_samples_split":trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features":     trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5]),
        }

    if name == "extra_trees":
        return {
            "n_estimators":     trial.suggest_int("n_estimators", 100, 800),
            "max_depth":        trial.suggest_int("max_depth", 5, 30),
            "min_samples_split":trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features":     trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5]),
        }

    if name == "xgboost":
        return {
            "n_estimators":    trial.suggest_int("n_estimators", 100, 1000),
            "max_depth":       trial.suggest_int("max_depth", 3, 10),
            "learning_rate":   trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "subsample":       trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight":trial.suggest_int("min_child_weight", 1, 10),
            "gamma":           trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha":       trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda":      trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }

    if name == "lightgbm":
        return {
            "n_estimators":    trial.suggest_int("n_estimators", 100, 1000),
            "max_depth":       trial.suggest_int("max_depth", 3, 12),
            "learning_rate":   trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "num_leaves":      trial.suggest_int("num_leaves", 20, 300),
            "subsample":       trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_samples":trial.suggest_int("min_child_samples", 5, 100),
            "reg_alpha":       trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda":      trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }

    raise ValueError(f"Unknown model: {name}")


# ── Model builder from param dict ─────────────────────────────────────────────

def _build_model(name: str, params: dict):
    """Instantiate an unfitted model from a param dict."""
    if name == "logistic_regression":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(
            **{k: params[k] for k in ["C", "solver", "max_iter"]},
            random_state=SEED, n_jobs=-1, class_weight="balanced",
        )

    if name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(
            **params, random_state=SEED, n_jobs=-1, class_weight="balanced",
        )

    if name == "extra_trees":
        from sklearn.ensemble import ExtraTreesClassifier
        return ExtraTreesClassifier(
            **params, random_state=SEED, n_jobs=-1, class_weight="balanced",
        )

    if name == "xgboost":
        try:
            from xgboost import XGBClassifier
            return XGBClassifier(
                **params,
                objective="multi:softprob", num_class=3,
                eval_metric="mlogloss", random_state=SEED,
                n_jobs=-1, verbosity=0, tree_method="hist",
            )
        except ImportError:
            from sklearn.ensemble import GradientBoostingClassifier
            safe = {k: params[k] for k in ["n_estimators", "max_depth", "learning_rate"]
                    if k in params}
            return GradientBoostingClassifier(**safe, random_state=SEED)

    if name == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
            return LGBMClassifier(
                **params,
                objective="multiclass", num_class=3,
                random_state=SEED, n_jobs=-1, verbose=-1,
            )
        except ImportError:
            from sklearn.ensemble import HistGradientBoostingClassifier
            safe = {k: params[k] for k in ["n_estimators", "max_depth", "learning_rate"]
                    if k in params and k != "num_leaves"}
            return HistGradientBoostingClassifier(**safe, random_state=SEED)

    raise ValueError(f"Unknown model: {name}")


# ── Cross-validated scoring ────────────────────────────────────────────────────

def _cv_score(
    name: str,
    params: dict,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    n_folds: int = 3,
) -> float:
    """3-fold stratified CV macro-F1 on the training set."""
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import f1_score
    from sklearn.utils.class_weight import compute_sample_weight

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    scores = []

    for fold_train_idx, fold_val_idx in skf.split(X_train, y_train):
        Xtr, Xval = X_train.iloc[fold_train_idx], X_train.iloc[fold_val_idx]
        ytr, yval = y_train[fold_train_idx],       y_train[fold_val_idx]

        m = _build_model(name, params)

        fit_kwargs: dict = {}
        if "xgboost" in type(m).__module__:
            sw = compute_sample_weight("balanced", ytr)
            fit_kwargs = {"sample_weight": sw, "eval_set": [(Xval.values, yval)], "verbose": False}
        elif "lightgbm" in type(m).__module__:
            sw = compute_sample_weight("balanced", ytr)
            fit_kwargs = {"sample_weight": sw, "eval_set": [(Xval, yval)], "callbacks": []}

        m.fit(Xtr, ytr, **fit_kwargs)
        yp = m.predict(Xval)
        scores.append(f1_score(yval, yp, average="macro", zero_division=0))

    return float(np.mean(scores))


# ── Single-model Optuna study ─────────────────────────────────────────────────

def tune_model(
    name: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_valid: pd.DataFrame,
    y_valid: np.ndarray,
    n_trials: int = 50,
    timeout:  int = 300,
    fresh:    bool = False,
    use_cv:   bool = True,
    n_folds:  int = 3,
) -> tuple[dict, list[dict]]:
    """
    Run an Optuna study for one model.

    Returns
    -------
    best_params : dict
    trial_history : list of {trial, value, params}
    """
    import optuna
    from optuna.pruners   import MedianPruner
    from optuna.samplers  import TPESampler
    from sklearn.metrics  import f1_score
    from sklearn.utils.class_weight import compute_sample_weight

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    STUDY_DIR.mkdir(parents=True, exist_ok=True)
    db_path = STUDY_DIR / f"hpo_{name}.db"
    if fresh and db_path.exists():
        db_path.unlink()
        logger.info(f"  [{name}] Deleted existing study DB.")

    storage  = f"sqlite:///{db_path}"
    study_name = f"hpo_{name}"

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        sampler=TPESampler(seed=SEED, n_startup_trials=10),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=1),
        load_if_exists=not fresh,
    )

    # Pre-import heavy optional libraries once (avoids per-trial import failures)
    _XGB  = None
    _LGBM = None
    try:
        from xgboost  import XGBClassifier  as _XGB   # type: ignore
    except ImportError:
        pass
    try:
        from lightgbm import LGBMClassifier as _LGBM  # type: ignore
    except ImportError:
        pass

    def objective(trial):
        params = _suggest_params(trial, name)

        if use_cv:
            # Cross-validated score (more robust, slower)
            score = _cv_score(name, params, X_train, y_train, n_folds=n_folds)
        else:
            # Single validation fold (faster)
            m = _build_model(name, params)
            fit_kwargs: dict = {}
            if _XGB is not None and name == "xgboost":
                sw = compute_sample_weight("balanced", y_train)
                fit_kwargs = {"sample_weight": sw,
                              "eval_set": [(X_valid.values, y_valid)], "verbose": False}
            elif _LGBM is not None and name == "lightgbm":
                sw = compute_sample_weight("balanced", y_train)
                fit_kwargs = {"sample_weight": sw,
                              "eval_set": [(X_valid, y_valid)], "callbacks": []}
            m.fit(X_train, y_train, **fit_kwargs)
            yp = m.predict(X_valid)
            score = f1_score(y_valid, yp, average="macro", zero_division=0)

        return score

    t0 = time.time()
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)
    elapsed = time.time() - t0

    best  = study.best_params
    bval  = study.best_value
    logger.info(f"  [{name}] Best f1_macro={bval:.4f}  trials={len(study.trials)}  "
                f"time={elapsed:.1f}s")
    logger.info(f"  [{name}] Best params: {best}")

    trial_history = [
        {"trial": t.number, "value": t.value, "params": t.params, "state": str(t.state)}
        for t in study.trials if t.value is not None
    ]
    return best, trial_history


# ── Convergence plot ───────────────────────────────────────────────────────────

def _plot_convergence(
    trial_history: list[dict],
    model_name:    str,
    save_path:     Path,
) -> None:
    if not trial_history:
        return
    trials = [t["trial"] for t in trial_history]
    values = [t["value"] for t in trial_history]
    best_so_far = [max(values[:i+1]) for i in range(len(values))]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(trials, values, s=15, alpha=0.5, color="#90CAF9", label="Trial f1")
    ax.plot(trials, best_so_far, color="#1565C0", linewidth=2, label="Best so far")
    ax.set_xlabel("Trial #")
    ax.set_ylabel("CV macro-F1")
    ax.set_title(f"HPO Convergence: {model_name}")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


# ── Load feature-engineered data ──────────────────────────────────────────────

def _load_processed_features(
    from_scratch: bool = False,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    """
    Load feature-selected X_train, y_train, X_valid, y_valid.

    If processed splits don't exist, runs the first 8 pipeline steps inline.
    """
    from src.data.schema import TARGET_COL
    train_feat = ROOT / "data" / "processed" / "train_features.parquet"
    valid_feat = ROOT / "data" / "processed" / "valid_features.parquet"

    if not from_scratch and train_feat.exists() and valid_feat.exists():
        logger.info("Loading pre-computed feature matrices from data/processed/")
        train_df = pd.read_parquet(train_feat)
        valid_df = pd.read_parquet(valid_feat)
        y_train = train_df[TARGET_COL].values if TARGET_COL in train_df.columns else None
        y_valid = valid_df[TARGET_COL].values if TARGET_COL in valid_df.columns else None
        feat_cols = [c for c in train_df.columns if c != TARGET_COL]
        return train_df[feat_cols], y_train, valid_df[feat_cols], y_valid

    # Fall back: run preprocessing inline (steps 1-8 of training_pipeline)
    logger.info("Feature matrices not found -- running preprocessing inline (steps 1-8).")
    from src.data.ingestion     import load_raw
    from src.data.validation    import validate_raw
    from src.data.split         import split_train_valid_test, save_splits
    from src.data.preprocessing import preprocess
    from src.features.imputers  import impute_missing
    from src.features.encoders  import encode_features
    from src.features.build_features import build_features
    from src.features.selectors import select_features
    from src.data.preprocessing import encode_target

    train_raw_full, _ = load_raw()
    validate_raw(train_raw_full, split="train")
    train_raw, valid_raw, test_raw = split_train_valid_test(
        train_raw_full, train_size=0.70, valid_size=0.15, test_size=0.15,
        random_state=SEED,
    )
    save_splits(train_raw, valid_raw, test_raw)

    train_clean, valid_clean, test_clean, _ = preprocess(train_raw, valid_raw, test_raw)

    y_train = encode_target(train_clean[TARGET_COL]).values
    y_valid = encode_target(valid_clean[TARGET_COL]).values
    train_clean = train_clean.drop(columns=[TARGET_COL])
    valid_clean = valid_clean.drop(columns=[TARGET_COL])
    test_clean  = test_clean.drop(columns=[TARGET_COL])

    train_imp, valid_imp, test_imp, imputer = impute_missing(
        train_clean, valid_clean, test_clean)
    train_enc, valid_enc, test_enc, encoder = encode_features(
        train_imp, valid_imp, test_imp)

    train_feat_df = build_features(train_enc)
    valid_feat_df = build_features(valid_enc)

    X_train, X_valid, _, selector = select_features(
        train_feat_df, valid_feat_df, build_features(test_imp),
        y_train, feature_config=None,
    )

    # Save for future runs
    (ROOT / "data" / "processed").mkdir(parents=True, exist_ok=True)
    train_out = X_train.copy()
    train_out[TARGET_COL] = y_train
    valid_out = X_valid.copy()
    valid_out[TARGET_COL] = y_valid
    train_out.to_parquet(train_feat, index=False)
    valid_out.to_parquet(valid_feat, index=False)
    logger.info("Feature matrices saved to data/processed/")

    return X_train, y_train, X_valid, y_valid


# ── Main HPO pipeline ─────────────────────────────────────────────────────────

def run_hpo_pipeline(
    models:   list[str] | None = None,
    n_trials: int  = 50,
    timeout:  int  = 300,
    fresh:    bool = False,
    use_cv:   bool = True,
    n_folds:  int  = 3,
) -> dict:
    """
    Run HPO for the given model list and return best params dict.

    Parameters
    ----------
    models   : which models to tune; defaults to ALL_MODELS
    n_trials : Optuna trial budget per model
    timeout  : wall-clock seconds per model study
    fresh    : delete existing SQLite studies and restart
    use_cv   : use k-fold CV inside objective (more robust, slower)
    n_folds  : number of CV folds when use_cv=True
    """
    t_total = time.time()
    models  = models or ALL_MODELS
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    STUDY_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("== HYPERPARAMETER TUNING PIPELINE")
    logger.info(f"   Models   : {models}")
    logger.info(f"   n_trials : {n_trials}  timeout : {timeout}s  use_cv : {use_cv}")
    logger.info("=" * 60)

    # ── Load data ────────────────────────────────────────────────────────────
    logger.info("\n-- Loading feature matrices --")
    X_train, y_train, X_valid, y_valid = _load_processed_features()
    logger.info(f"   X_train {X_train.shape}   X_valid {X_valid.shape}")

    # ── Load existing best params (may be partially populated) ───────────────
    best_params: dict = {}
    if BEST_PARAMS_PATH.exists():
        with open(BEST_PARAMS_PATH) as f:
            best_params = json.load(f)
        logger.info(f"   Loaded existing best params for: {list(best_params.keys())}")

    all_histories: dict = {}

    # ── Tune each model ──────────────────────────────────────────────────────
    for model_name in models:
        logger.info(f"\n-- Tuning: {model_name} --")
        params, history = tune_model(
            name=model_name,
            X_train=X_train, y_train=y_train,
            X_valid=X_valid, y_valid=y_valid,
            n_trials=n_trials,
            timeout=timeout,
            fresh=fresh,
            use_cv=use_cv,
            n_folds=n_folds,
        )
        best_params[model_name] = params
        all_histories[model_name] = history

        # Convergence plot
        plot_path = REPORT_DIR / f"hpo_{model_name}.png"
        _plot_convergence(history, model_name, plot_path)
        logger.info(f"  [{model_name}] Convergence plot -> {plot_path}")

    # ── Save results ─────────────────────────────────────────────────────────
    with open(BEST_PARAMS_PATH, "w") as f:
        json.dump(best_params, f, indent=2)
    logger.info(f"\n-- Best params saved -> {BEST_PARAMS_PATH}")

    full_results = {
        "n_trials": n_trials,
        "timeout":  timeout,
        "use_cv":   use_cv,
        "n_folds":  n_folds,
        "models":   models,
        "best_params":  best_params,
        "trial_histories": all_histories,
    }
    with open(FULL_RESULTS_PATH, "w") as f:
        json.dump(full_results, f, indent=2)
    logger.info(f"-- Full results saved -> {FULL_RESULTS_PATH}")

    # ── Summary table ────────────────────────────────────────────────────────
    logger.info("\n-- HPO Summary (best CV f1_macro per model) --")
    for mname, history in all_histories.items():
        if history:
            best_val = max(h["value"] for h in history if h["value"] is not None)
            n_completed = sum(1 for h in history if h["value"] is not None)
            logger.info(f"   {mname:<22} f1={best_val:.4f}  completed_trials={n_completed}")

    elapsed_total = time.time() - t_total
    logger.info(f"\n== HPO COMPLETE in {elapsed_total:.1f}s ==")
    logger.info("   Run training_pipeline.py to use these params for the full training run.")

    return best_params


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter tuning pipeline")
    parser.add_argument(
        "--model", nargs="+", default=None,
        choices=ALL_MODELS + ["all"],
        help="Models to tune (default: all). E.g. --model lightgbm xgboost",
    )
    parser.add_argument("--n-trials", type=int, default=50, help="Optuna trials per model")
    parser.add_argument("--timeout",  type=int, default=300, help="Seconds per model study")
    parser.add_argument("--fresh",    action="store_true",   help="Delete existing studies")
    parser.add_argument("--no-cv",    action="store_true",   help="Use single val fold, no CV")
    parser.add_argument("--n-folds",  type=int, default=3,   help="CV folds (default 3)")
    args = parser.parse_args()

    models = None
    if args.model and "all" not in args.model:
        models = args.model

    run_hpo_pipeline(
        models=models,
        n_trials=args.n_trials,
        timeout=args.timeout,
        fresh=args.fresh,
        use_cv=not args.no_cv,
        n_folds=args.n_folds,
    )
