"""
src/pipelines/training_pipeline.py
====================================
Full end-to-end training pipeline.

Steps
-----
1.  Ingest  -- load train_raw.csv only
2.  Validate
3.  Split   -- 70 / 15 / 15 stratified
4.  Preprocess (clean + cap outliers)
5.  Impute
6.  Encode
7.  Build features
8.  Select features
9.  Train 5 candidate models
10. Rank all 5 -> save model_ranking.csv
11. Light Optuna tuning on top-3
12. Build 1 ensemble from tuned top-3
13. Evaluate all (5 raw + 3 tuned + ensemble) on validation
14. Select final production model
15. Calibrate final model
16. Explainability (feature importance + SHAP)
17. Fairness diagnostics
18. Drift reference generation
19. Power BI exports
20. Serialize final bundle

Usage
-----
    python -m src.pipelines.training_pipeline
    python src/pipelines/training_pipeline.py
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# ── Data ──────────────────────────────────────────────────────────────────────
from src.data.ingestion     import load_raw
from src.data.validation    import validate_raw
from src.data.split         import split_train_valid_test, save_splits
from src.data.preprocessing import preprocess
from src.data.schema        import TARGET_COL

# ── Features ──────────────────────────────────────────────────────────────────
from src.features.imputers       import impute_missing
from src.features.encoders       import encode_features
from src.features.build_features import build_features
from src.features.selectors      import select_features

# ── Models ────────────────────────────────────────────────────────────────────
from src.models.evaluate  import evaluate, save_report
from src.models.calibrate import calibrate_model, calibration_report, plot_calibration_curve
from src.models.registry  import register_model, promote_to_production
from src.models.serialize import ModelBundle, save_bundle

# ── Risk ──────────────────────────────────────────────────────────────────────
from src.risk.explainability import global_feature_importance, plot_feature_importance
from src.risk.fairness       import fairness_report, fairness_summary

# ── Monitoring / reports ──────────────────────────────────────────────────────
from src.monitoring.dashboard_metrics import export_all_powerbi_tables
from src.monitoring.input_monitor     import monitor_input, save_input_monitor_report

# ── Utils ─────────────────────────────────────────────────────────────────────
from src.utils.io     import write_csv, write_json, write_parquet
from src.utils.logger import get_logger

logger = get_logger(__name__)

REPORT_DIR    = ROOT / "artifacts" / "reports"
MODEL_DIR     = ROOT / "artifacts" / "models"
PRED_DIR      = ROOT / "artifacts" / "predictions"
DRIFT_DIR     = ROOT / "artifacts" / "drift_reports"
REF_DIR       = ROOT / "data" / "reference"
SEED          = 42
CLASS_LABELS  = ["Poor", "Standard", "Good"]
LABEL_MAP     = {0: "Poor", 1: "Standard", 2: "Good"}


# ─────────────────────────────────────────────────────────────────────────────
# Model factory
# ─────────────────────────────────────────────────────────────────────────────

def _build_models() -> dict:
    """Return 5 untrained sklearn-compatible classifiers."""
    from sklearn.linear_model  import LogisticRegression
    from sklearn.ensemble      import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
    from sklearn.utils.class_weight import compute_class_weight

    models: dict = {}

    # 1. Logistic Regression -- interpretable baseline
    # multi_class removed in sklearn >= 1.7 (always auto for lbfgs)
    models["logistic_regression"] = LogisticRegression(
        max_iter=1000, C=1.0, solver="lbfgs",
        random_state=SEED, n_jobs=-1, class_weight="balanced",
    )

    # 2. Random Forest
    models["random_forest"] = RandomForestClassifier(
        n_estimators=200, max_depth=12, min_samples_split=5,
        random_state=SEED, n_jobs=-1, class_weight="balanced",
    )

    # 3. Extra Trees
    models["extra_trees"] = ExtraTreesClassifier(
        n_estimators=200, max_depth=12, min_samples_split=5,
        random_state=SEED, n_jobs=-1, class_weight="balanced",
    )

    # 4. XGBoost
    try:
        from xgboost import XGBClassifier
        models["xgboost"] = XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            objective="multi:softprob", num_class=3,
            eval_metric="mlogloss", random_state=SEED,
            n_jobs=-1, verbosity=0, tree_method="hist",
        )
    except ImportError:
        from sklearn.ensemble import GradientBoostingClassifier
        models["xgboost"] = GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05, random_state=SEED,
        )

    # 5. LightGBM
    try:
        from lightgbm import LGBMClassifier
        models["lightgbm"] = LGBMClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            num_leaves=63, subsample=0.8, colsample_bytree=0.8,
            objective="multiclass", num_class=3,
            random_state=SEED, n_jobs=-1, verbose=-1,
            # No class_weight: training code applies sample_weight instead
        )
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier
        models["lightgbm"] = HistGradientBoostingClassifier(
            max_iter=300, max_depth=6, learning_rate=0.05, random_state=SEED,
        )

    return models


# ─────────────────────────────────────────────────────────────────────────────
# Train & evaluate a single model
# ─────────────────────────────────────────────────────────────────────────────

def _train_evaluate(
    name: str,
    model,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_valid: pd.DataFrame,
    y_valid: np.ndarray,
) -> tuple[object, dict]:
    """Fit one model, evaluate on validation, return (fitted_model, metrics)."""
    from sklearn.utils.class_weight import compute_sample_weight

    logger.info(f"  Training: {name} ...")
    t0 = time.time()

    fit_kwargs: dict = {}

    # XGBoost supports sample_weight + eval_set
    if "xgboost" in type(model).__module__:
        sw = compute_sample_weight("balanced", y_train)
        fit_kwargs = {
            "sample_weight": sw,
            "eval_set": [(X_valid.values, y_valid)],
            "verbose": False,
        }
        model.set_params(early_stopping_rounds=30)
    elif "lightgbm" in type(model).__module__:
        sw = compute_sample_weight("balanced", y_train)
        fit_kwargs = {
            "sample_weight": sw,
            "eval_set": [(X_valid, y_valid)],
            "callbacks": [],
        }
    else:
        # sklearn estimators that accept class_weight handle balance internally
        pass

    try:
        model.fit(X_train, y_train, **fit_kwargs)
    except TypeError:
        model.fit(X_train, y_train)

    elapsed = round(time.time() - t0, 2)

    y_prob = model.predict_proba(X_valid)
    y_pred = np.argmax(y_prob, axis=1)

    from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
    metrics = {
        "model_name":      name,
        "train_time_sec":  elapsed,
        "f1_macro":        round(f1_score(y_valid, y_pred, average="macro",    zero_division=0), 4),
        "f1_weighted":     round(f1_score(y_valid, y_pred, average="weighted", zero_division=0), 4),
        "accuracy":        round(accuracy_score(y_valid, y_pred), 4),
        "precision_macro": round(precision_score(y_valid, y_pred, average="macro", zero_division=0), 4),
        "recall_macro":    round(recall_score(y_valid, y_pred, average="macro",    zero_division=0), 4),
        "auc_ovr":         round(roc_auc_score(y_valid, y_prob, multi_class="ovr", average="macro"), 4),
    }
    logger.info(f"    {name}: f1_macro={metrics['f1_macro']:.4f}  acc={metrics['accuracy']:.4f}"
                f"  [{elapsed}s]")
    return model, metrics


# ─────────────────────────────────────────────────────────────────────────────
# Rank models
# ─────────────────────────────────────────────────────────────────────────────

def _rank_models(all_metrics: list[dict]) -> pd.DataFrame:
    """Rank models by f1_macro -> f1_weighted -> accuracy."""
    df = pd.DataFrame(all_metrics)
    df = df.sort_values(
        ["f1_macro", "f1_weighted", "accuracy"],
        ascending=False,
    ).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Optuna tuning for a single model
# ─────────────────────────────────────────────────────────────────────────────

def _tune_model(
    name: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_valid: pd.DataFrame,
    y_valid: np.ndarray,
    n_trials: int = 20,
    timeout:  int = 180,
) -> tuple[object, dict, dict]:
    """
    Light Optuna tuning for the given model name.
    Returns (best_model, best_params, best_metrics).
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    from sklearn.metrics import f1_score
    from sklearn.utils.class_weight import compute_sample_weight

    def _f1(model, X, y):
        yp = model.predict(X)
        return f1_score(y, yp, average="macro", zero_division=0)

    # Pre-import optional libraries ONCE outside the objective to avoid
    # per-trial import failures (Windows DLL / module-cache issues).
    _XGBClassifier = None
    _LGBMClassifier = None
    try:
        from xgboost import XGBClassifier as _XGBClassifier  # type: ignore
    except ImportError:
        pass
    try:
        from lightgbm import LGBMClassifier as _LGBMClassifier  # type: ignore
    except ImportError:
        pass

    best_model   = [None]
    best_params  = [{}]
    best_metric  = [-1.0]

    def objective(trial):
        if name == "logistic_regression":
            from sklearn.linear_model import LogisticRegression
            C = trial.suggest_float("C", 0.001, 100.0, log=True)
            m = LogisticRegression(C=C, max_iter=1000, solver="lbfgs",
                                   random_state=SEED,
                                   n_jobs=-1, class_weight="balanced")
            m.fit(X_train, y_train)

        elif name == "random_forest":
            from sklearn.ensemble import RandomForestClassifier
            m = RandomForestClassifier(
                n_estimators = trial.suggest_int("n_estimators", 100, 400),
                max_depth    = trial.suggest_int("max_depth", 6, 20),
                min_samples_split = trial.suggest_int("min_samples_split", 2, 10),
                random_state=SEED, n_jobs=-1, class_weight="balanced",
            )
            m.fit(X_train, y_train)

        elif name == "extra_trees":
            from sklearn.ensemble import ExtraTreesClassifier
            m = ExtraTreesClassifier(
                n_estimators = trial.suggest_int("n_estimators", 100, 400),
                max_depth    = trial.suggest_int("max_depth", 6, 20),
                min_samples_split = trial.suggest_int("min_samples_split", 2, 10),
                random_state=SEED, n_jobs=-1, class_weight="balanced",
            )
            m.fit(X_train, y_train)

        elif name == "xgboost":
            sw = compute_sample_weight("balanced", y_train)
            if _XGBClassifier is not None:
                m = _XGBClassifier(
                    n_estimators   = trial.suggest_int("n_estimators", 100, 500),
                    max_depth      = trial.suggest_int("max_depth", 3, 8),
                    learning_rate  = trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                    subsample      = trial.suggest_float("subsample", 0.6, 1.0),
                    colsample_bytree = trial.suggest_float("colsample_bytree", 0.6, 1.0),
                    objective="multi:softprob", num_class=3,
                    eval_metric="mlogloss", random_state=SEED, n_jobs=-1,
                    verbosity=0, tree_method="hist",
                )
                m.fit(X_train, y_train, sample_weight=sw,
                      eval_set=[(X_valid.values, y_valid)], verbose=False)
            else:
                from sklearn.ensemble import GradientBoostingClassifier
                m = GradientBoostingClassifier(
                    n_estimators = trial.suggest_int("n_estimators", 100, 300),
                    max_depth    = trial.suggest_int("max_depth", 3, 6),
                    learning_rate= trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                    random_state=SEED,
                )
                m.fit(X_train, y_train)

        elif name == "lightgbm":
            sw = compute_sample_weight("balanced", y_train)
            if _LGBMClassifier is not None:
                m = _LGBMClassifier(
                    n_estimators    = trial.suggest_int("n_estimators", 100, 500),
                    max_depth       = trial.suggest_int("max_depth", 4, 10),
                    learning_rate   = trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                    num_leaves      = trial.suggest_int("num_leaves", 20, 100),
                    subsample       = trial.suggest_float("subsample", 0.6, 1.0),
                    colsample_bytree= trial.suggest_float("colsample_bytree", 0.6, 1.0),
                    objective="multiclass", num_class=3, random_state=SEED,
                    n_jobs=-1, verbose=-1,
                )
                m.fit(X_train, y_train, sample_weight=sw,
                      eval_set=[(X_valid, y_valid)], callbacks=[])
            else:
                from sklearn.ensemble import HistGradientBoostingClassifier
                m = HistGradientBoostingClassifier(
                    max_iter     = trial.suggest_int("max_iter", 100, 400),
                    max_depth    = trial.suggest_int("max_depth", 4, 10),
                    learning_rate= trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                    random_state=SEED,
                )
                m.fit(X_train, y_train)
        else:
            raise ValueError(f"Unknown model name for tuning: {name}")

        score = _f1(m, X_valid, y_valid)
        if score > best_metric[0]:
            best_metric[0] = score
            best_model[0]  = m
            best_params[0] = trial.params
        return score

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)

    model    = best_model[0]
    params   = best_params[0]

    y_prob   = model.predict_proba(X_valid)
    y_pred   = np.argmax(y_prob, axis=1)

    from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
    metrics = {
        "model_name":      name,
        "f1_macro":        round(f1_score(y_valid, y_pred, average="macro",    zero_division=0), 4),
        "f1_weighted":     round(f1_score(y_valid, y_pred, average="weighted", zero_division=0), 4),
        "accuracy":        round(accuracy_score(y_valid, y_pred), 4),
        "precision_macro": round(precision_score(y_valid, y_pred, average="macro", zero_division=0), 4),
        "recall_macro":    round(recall_score(y_valid, y_pred, average="macro",    zero_division=0), 4),
        "auc_ovr":         round(roc_auc_score(y_valid, y_prob, multi_class="ovr", average="macro"), 4),
        "best_params":     params,
    }
    logger.info(f"  Tuned {name}: f1_macro={metrics['f1_macro']:.4f}  params={params}")
    return model, params, metrics


# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# Module-level ensemble class (must be at module level for joblib to pickle it)
# ─────────────────────────────────────────────────────────────────────────────

class SoftVotingEnsemble:
    """
    Picklable soft-voting ensemble over a dict of fitted classifiers.
    Must be defined at module level (not inside a function) for joblib pickling.
    """

    def __init__(self, models: dict):
        self._models = models

    # sklearn clone() compatibility
    def get_params(self, deep: bool = True) -> dict:
        return {"models": self._models}

    def set_params(self, **params) -> "SoftVotingEnsemble":
        if "models" in params:
            self._models = params["models"]
        return self

    def fit(self, X, y, **kwargs):
        """No-op: base models are already fitted. Required for sklearn wrappers."""
        return self

    def predict_proba(self, X) -> np.ndarray:
        return np.mean([m.predict_proba(X) for m in self._models.values()], axis=0)

    def predict(self, X) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)

    @property
    def classes_(self) -> np.ndarray:
        return np.array([0, 1, 2])


# Build ensemble from 3 tuned models
# ─────────────────────────────────────────────────────────────────────────────

def _build_ensemble(
    tuned_models: dict,
    X_valid: pd.DataFrame,
    y_valid: np.ndarray,
) -> tuple[object, dict]:
    """Soft-voting ensemble of the 3 tuned models."""
    from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score

    logger.info("  Building soft-voting ensemble ...")
    probas = np.mean(
        [m.predict_proba(X_valid) for _, m in tuned_models.items()], axis=0
    )
    y_pred = np.argmax(probas, axis=1)

    metrics = {
        "model_name":      "ensemble_soft_voting",
        "f1_macro":        round(f1_score(y_valid, y_pred, average="macro",    zero_division=0), 4),
        "f1_weighted":     round(f1_score(y_valid, y_pred, average="weighted", zero_division=0), 4),
        "accuracy":        round(accuracy_score(y_valid, y_pred), 4),
        "precision_macro": round(precision_score(y_valid, y_pred, average="macro", zero_division=0), 4),
        "recall_macro":    round(recall_score(y_valid, y_pred, average="macro",    zero_division=0), 4),
        "auc_ovr":         round(roc_auc_score(y_valid, probas, multi_class="ovr", average="macro"), 4),
    }
    logger.info(f"  Ensemble: f1_macro={metrics['f1_macro']:.4f}  acc={metrics['accuracy']:.4f}")

    soft_ensemble = SoftVotingEnsemble(tuned_models)
    return soft_ensemble, metrics


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_training_pipeline() -> ModelBundle:
    t_start = time.time()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    DRIFT_DIR.mkdir(parents=True, exist_ok=True)
    REF_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. INGEST ─────────────────────────────────────────────────────────────
    logger.info("\n== 1. INGEST ==========================================")
    train_raw_full, _ = load_raw()   # Only use train_raw.csv
    logger.info(f"Loaded train_raw: {train_raw_full.shape}")

    # ── 2. VALIDATE ───────────────────────────────────────────────────────────
    logger.info("\n== 2. VALIDATE ========================================")
    validate_raw(train_raw_full, split="train")

    # ── 3. SPLIT 70/15/15 ────────────────────────────────────────────────────
    logger.info("\n== 3. SPLIT (70/15/15) ================================")
    train_raw, valid_raw, test_raw = split_train_valid_test(
        train_raw_full, train_size=0.70, valid_size=0.15, test_size=0.15,
        random_state=SEED,
    )
    save_splits(train_raw, valid_raw, test_raw)

    # ── 4. PREPROCESS ─────────────────────────────────────────────────────────
    logger.info("\n== 4. PREPROCESS ======================================")
    from src.data.preprocessing import encode_target
    train_clean, valid_clean, test_clean, annual_income_cap = preprocess(train_raw, valid_raw, test_raw)
    # All three splits come from labelled train_raw, so encode target for test too
    test_clean = encode_target(test_clean)

    # ── 5. IMPUTE ─────────────────────────────────────────────────────────────
    logger.info("\n== 5. IMPUTE ==========================================")
    imputer, train_imp, valid_imp, test_imp = impute_missing(
        train_clean, valid_clean, test_clean
    )

    # ── 6. ENCODE ─────────────────────────────────────────────────────────────
    logger.info("\n== 6. ENCODE ==========================================")
    encoder, train_enc, valid_enc, test_enc = encode_features(
        train_imp, valid_imp, test_imp
    )

    # ── 7. BUILD FEATURES ─────────────────────────────────────────────────────
    logger.info("\n== 7. BUILD FEATURES ==================================")
    train_feat = build_features(train_enc)
    valid_feat = build_features(valid_enc)
    test_feat  = build_features(test_enc)

    # ── 8. SELECT FEATURES ────────────────────────────────────────────────────
    logger.info("\n== 8. SELECT FEATURES =================================")
    selector, train_sel, valid_sel, test_sel = select_features(
        train_feat, valid_feat, test_feat
    )

    X_train = train_sel.drop(columns=[TARGET_COL], errors="ignore")
    y_train = train_sel[TARGET_COL].astype(int).values
    X_valid = valid_sel.drop(columns=[TARGET_COL], errors="ignore")
    y_valid = valid_sel[TARGET_COL].astype(int).values
    X_test  = test_sel.drop(columns=[TARGET_COL], errors="ignore")
    y_test  = test_sel[TARGET_COL].astype(int).values

    # Final NaN fill: derived ratio features can produce NaN (div-by-zero).
    # Fill with train column medians computed here (no leakage since fitted on train).
    _train_medians = X_train.median()
    X_train = X_train.fillna(_train_medians)
    X_valid = X_valid.fillna(_train_medians)
    X_test  = X_test.fillna(_train_medians)

    nan_train = X_train.isna().sum().sum()
    if nan_train > 0:
        logger.warning(f"  {nan_train} NaN remaining after fill -- filling with 0.")
        X_train = X_train.fillna(0)
        X_valid = X_valid.fillna(0)
        X_test  = X_test.fillna(0)

    logger.info(f"X_train {X_train.shape}  X_valid {X_valid.shape}  X_test {X_test.shape}")

    # ── 9. TRAIN 5 CANDIDATE MODELS ───────────────────────────────────────────
    logger.info("\n== 9. TRAIN 5 MODELS ==================================")
    candidate_models = _build_models()
    raw_results:  list[dict]  = []
    fitted_models: dict[str, object] = {}

    for name, model in candidate_models.items():
        fitted, metrics = _train_evaluate(name, model, X_train, y_train, X_valid, y_valid)
        raw_results.append(metrics)
        fitted_models[name] = fitted

    # ── 10. RANK ALL 5 ────────────────────────────────────────────────────────
    logger.info("\n== 10. RANK MODELS ====================================")
    ranking_df = _rank_models(raw_results)
    logger.info(f"\n{ranking_df[['rank','model_name','f1_macro','f1_weighted','accuracy']].to_string(index=False)}")

    ranking_path = REPORT_DIR / "model_ranking.csv"
    write_csv(ranking_df, ranking_path)

    ranking_md_rows = ["# Model Ranking\n",
                       ranking_df[["rank","model_name","f1_macro","f1_weighted","accuracy","auc_ovr"]].to_string(index=False),
                       "\n\n## Selection Rationale",
                       f"\nTop-3 selected by primary metric (macro F1): "
                       f"{', '.join(ranking_df.head(3)['model_name'].tolist())}",
                       "\nMacro F1 is preferred because this is a multiclass problem with class imbalance."]
    (REPORT_DIR / "model_ranking.md").write_text("\n".join(ranking_md_rows))

    top3_names = ranking_df.head(3)["model_name"].tolist()
    logger.info(f"Top-3: {top3_names}")

    # Save 5-model comparison CSV (Power BI)
    raw_df = pd.DataFrame(raw_results)
    write_csv(raw_df, REPORT_DIR / "raw_model_comparison.csv")

    # ── 11. TUNE TOP-3 ────────────────────────────────────────────────────────
    logger.info("\n== 11. TUNE TOP-3 =====================================")
    tuned_models:   dict[str, object] = {}
    tuning_results: list[dict]        = []
    best_params_all: dict             = {}

    for name in top3_names:
        logger.info(f"  Tuning: {name}")
        model, params, metrics = _tune_model(
            name, X_train, y_train, X_valid, y_valid,
            n_trials=20, timeout=120,
        )
        tuned_models[name]     = model
        tuning_results.append(metrics)
        best_params_all[name]  = params

    tuning_df = pd.DataFrame(tuning_results)
    write_csv(tuning_df, REPORT_DIR / "top3_tuning_results.csv")
    write_json(best_params_all, REPORT_DIR / "best_hyperparameters.json")
    logger.info(f"\n{tuning_df[['model_name','f1_macro','f1_weighted','accuracy']].to_string(index=False)}")

    # ── 12. BUILD ENSEMBLE ────────────────────────────────────────────────────
    logger.info("\n== 12. ENSEMBLE =======================================")
    ensemble, ensemble_metrics = _build_ensemble(tuned_models, X_valid, y_valid)

    # ── 13. FINAL COMPARISON ──────────────────────────────────────────────────
    logger.info("\n== 13. FINAL COMPARISON ===============================")
    comparison_rows = tuning_results + [ensemble_metrics]
    comparison_df   = pd.DataFrame(comparison_rows).sort_values("f1_macro", ascending=False)
    comparison_df.insert(0, "rank", range(1, len(comparison_df) + 1))
    write_csv(comparison_df, REPORT_DIR / "final_model_comparison.csv")
    logger.info(f"\n{comparison_df[['rank','model_name','f1_macro','f1_weighted','accuracy']].to_string(index=False)}")

    # ── 14. SELECT FINAL MODEL ────────────────────────────────────────────────
    logger.info("\n== 14. SELECT FINAL MODEL =============================")
    # Compare ensemble vs best tuned single model
    best_tuned_row     = comparison_df[comparison_df["model_name"] != "ensemble_soft_voting"].iloc[0]
    ensemble_row       = comparison_df[comparison_df["model_name"] == "ensemble_soft_voting"].iloc[0]
    best_tuned_name    = best_tuned_row["model_name"]
    best_tuned_f1      = best_tuned_row["f1_macro"]
    ensemble_f1        = ensemble_row["f1_macro"]

    if ensemble_f1 >= best_tuned_f1:
        final_model      = ensemble
        final_model_name = "ensemble_soft_voting"
        final_metrics    = ensemble_metrics
        justification    = (f"Ensemble (f1_macro={ensemble_f1:.4f}) >= "
                            f"best single model {best_tuned_name} ({best_tuned_f1:.4f}). "
                            "Ensemble provides better generalisation.")
    else:
        final_model      = tuned_models[best_tuned_name]
        final_model_name = best_tuned_name
        final_metrics    = best_tuned_row.to_dict()
        justification    = (f"Best tuned model {best_tuned_name} (f1_macro={best_tuned_f1:.4f}) "
                            f"> ensemble ({ensemble_f1:.4f}). "
                            "Single model preferred for simpler inference.")

    logger.info(f"Final model: {final_model_name} -- {justification}")
    write_json(
        {"final_model": final_model_name, "justification": justification, **final_metrics},
        REPORT_DIR / "final_model_selection.json",
    )

    # ── 15. FULL EVALUATION ON VALID + TEST ───────────────────────────────────
    logger.info("\n== 15. FULL EVALUATION ================================")
    for X, y, split_name in [(X_valid, y_valid, "valid"), (X_test, y_test, "test")]:
        y_prob_eval = final_model.predict_proba(X)
        y_pred_eval = np.argmax(y_prob_eval, axis=1)
        report      = evaluate(y, y_pred_eval, y_prob_eval, split=split_name)
        save_report(report, f"eval_{split_name}_final.json")

        # Confusion matrix plot
        _plot_confusion_matrix(y, y_pred_eval, split_name)

    # ── 16. CALIBRATE ─────────────────────────────────────────────────────────
    logger.info("\n== 16. CALIBRATE ======================================")
    y_prob_before = final_model.predict_proba(X_valid)
    try:
        cal_model  = calibrate_model(final_model, X_valid, y_valid, method="isotonic")
        y_prob_cal = cal_model.predict_proba(X_valid)
        cal_report = calibration_report(y_valid, y_prob_before, y_prob_cal)
        write_json(cal_report, REPORT_DIR / "calibration_report.json")
        plot_calibration_curve(y_valid, y_prob_before, y_prob_cal,
                               save_path=REPORT_DIR / "calibration_curve.png")
        serving_model = cal_model
        logger.info(f"  ECE before={cal_report['ece_before_mean']:.4f}  after={cal_report['ece_after_mean']:.4f}")
    except Exception as e:
        logger.warning(f"  Calibration failed ({e}) -- using uncalibrated model.")
        serving_model = final_model

    # ── 17. EXPLAINABILITY ────────────────────────────────────────────────────
    logger.info("\n== 17. EXPLAINABILITY =================================")
    # For explainability, prefer the best single tuned model (tree-based, supports feature_importances_)
    # The ensemble's built-in importance is not directly accessible.
    _expl_model_name = top3_names[0]  # best single model
    _expl_model      = tuned_models[_expl_model_name]
    try:
        X_sample = X_valid.sample(min(500, len(X_valid)), random_state=SEED)
        imp_df   = global_feature_importance(
            _expl_model, X_sample, method="shap", background_samples=100
        )
        write_csv(imp_df, REPORT_DIR / "shap_feature_importance.csv")
        write_csv(imp_df, REPORT_DIR / "feature_importance.csv")
        plot_feature_importance(imp_df, top_n=20,
                                save_path=REPORT_DIR / "feature_importance.png")
        logger.info(f"  SHAP importances computed from {_expl_model_name}.")
    except Exception as e:
        logger.warning(f"  SHAP failed ({e}) -- trying built-in importance.")
        try:
            imp_df = global_feature_importance(_expl_model, X_valid, method="builtin")
            write_csv(imp_df, REPORT_DIR / "feature_importance.csv")
            plot_feature_importance(imp_df, top_n=20,
                                    save_path=REPORT_DIR / "feature_importance.png")
            logger.info(f"  Built-in importances from {_expl_model_name}.")
        except Exception as e2:
            logger.warning(f"  Built-in importance also failed: {e2}")

    # ── 18. FAIRNESS ──────────────────────────────────────────────────────────
    logger.info("\n== 18. FAIRNESS =======================================")
    try:
        y_pred_valid = np.argmax(final_model.predict_proba(X_valid), axis=1)
        fair_df_input = valid_raw.copy().reset_index(drop=True)
        fair_df_input["y_true"] = y_valid
        fair_df_input["y_pred"] = y_pred_valid
        if TARGET_COL in fair_df_input.columns:
            fair_df_input["y_true"] = valid_raw[TARGET_COL].map(
                {"Poor": 0, "Standard": 1, "Good": 2}
            ).fillna(fair_df_input["y_true"]).astype(int)

        fair_df = fairness_report(fair_df_input, "y_true", "y_pred", group_col="Occupation")
        if not fair_df.empty:
            write_csv(fair_df, REPORT_DIR / "fairness_report.csv")
            summary = fairness_summary(fair_df)
            write_json(summary, REPORT_DIR / "fairness_summary.json")
            logger.info(f"  Fairness disparity gap: {summary.get('disparity_gap', 'N/A')}")
    except Exception as e:
        logger.warning(f"  Fairness analysis failed: {e}")

    # ── 19. REFERENCE DATA FOR DRIFT ─────────────────────────────────────────
    logger.info("\n== 19. REFERENCE DATA =================================")
    write_parquet(X_train, REF_DIR / "train_reference.parquet")
    write_json(
        {col: sorted(train_raw[col].dropna().unique().tolist())
         for col in ["Occupation", "Credit_Mix", "Payment_of_Min_Amount", "Payment_Behaviour"]
         if col in train_raw.columns},
        REF_DIR / "category_sets.json",
    )
    logger.info(f"  Reference saved -> {REF_DIR}")

    # ── 20. SAMPLE PREDICTIONS ────────────────────────────────────────────────
    logger.info("\n== 20. SAMPLE PREDICTIONS =============================")
    y_prob_test = final_model.predict_proba(X_test)
    y_pred_test = np.argmax(y_prob_test, axis=1)

    sample_df = X_test.head(200).copy().reset_index(drop=True)
    sample_df["y_true"]             = [LABEL_MAP[l] for l in y_test[:200]]
    sample_df["y_pred"]             = [LABEL_MAP[l] for l in y_pred_test[:200]]
    sample_df["correct"]            = sample_df["y_true"] == sample_df["y_pred"]
    for k, cls in enumerate(CLASS_LABELS):
        sample_df[f"prob_{cls}"] = y_prob_test[:200, k]
    write_csv(sample_df, PRED_DIR / "sample_predictions.csv")

    # ── 21. POWER BI EXPORTS ──────────────────────────────────────────────────
    logger.info("\n== 21. POWER BI EXPORTS ===============================")
    # Input quality on test split
    ref_stats = {"category_sets": json.loads(
        (REF_DIR / "category_sets.json").read_text()
    )}
    mon_result = monitor_input(test_raw, reference_stats=ref_stats, label="test_split")
    save_input_monitor_report(mon_result, REPORT_DIR / "powerbi_input_quality.csv")

    export_all_powerbi_tables(
        metrics      = final_metrics if isinstance(final_metrics, dict) else {},
        predictions  = y_pred_test,
        monitor_result = mon_result,
        model_version  = "1.0.0",
        out_dir        = REPORT_DIR,
    )

    # Drift summary Power BI export
    from src.monitoring.drift_monitor import run_drift_check
    run_drift_check(X_test, output_dir=DRIFT_DIR, label="test_split_check")

    # ── 22. SERIALIZE FINAL BUNDLE ────────────────────────────────────────────
    logger.info("\n== 22. SERIALIZE ======================================")
    metadata = {
        "model_version":       "1.0.0",
        "final_model_name":    final_model_name,
        "final_model_type":    type(final_model).__name__,
        "top3_models":         top3_names,
        "feature_names":       selector.feature_names_out,
        "annual_income_cap":   float(annual_income_cap),
        "n_train_rows":        len(y_train),
        "n_valid_rows":        len(y_valid),
        "n_test_rows":         len(y_test),
        "training_time_total": round(time.time() - t_start, 2),
        "justification":       justification,
        "metrics_valid":       final_metrics,
    }

    bundle = ModelBundle(
        model    = serving_model,
        imputer  = imputer,
        encoder  = encoder,
        selector = selector,
        metadata = metadata,
    )
    bundle_path = save_bundle(bundle, MODEL_DIR / "final_model_bundle.pkl")

    # Register in model registry
    register_model(
        model_name    = final_model_name,
        version       = "1.0.0",
        artifact_path = bundle_path,
        metrics       = final_metrics if isinstance(final_metrics, dict) else {},
        tags          = {"pipeline": "training_pipeline_v1"},
    )
    promote_to_production(final_model_name, "1.0.0")

    total_time = round(time.time() - t_start, 1)
    logger.info(f"\n== PIPELINE COMPLETE in {total_time}s ======================")
    logger.info(f"  Final model      : {final_model_name}")
    logger.info(f"  Bundle           : {bundle_path}")
    logger.info(f"  Reports dir      : {REPORT_DIR}")
    return bundle


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _plot_confusion_matrix(y_true, y_pred, split: str) -> None:
    """Plot and save confusion matrix."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

    cm  = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(cm, display_labels=CLASS_LABELS).plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix -- {split.capitalize()} Split", fontsize=14, fontweight="bold")
    plt.tight_layout()

    path = REPORT_DIR / f"confusion_matrix_{split}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Confusion matrix saved -> {path}")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_training_pipeline()
