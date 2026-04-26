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

# ruff: noqa: E402

import json
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

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
from src.models.ensemble  import SoftVotingEnsemble
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
TRAIN_SIZE    = 0.70
VALID_SIZE    = 0.15
TEST_SIZE     = 0.15


def _env_bool(name: str, default: bool = True) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _normalize_mlflow_tracking_uri(uri: str) -> str:
    raw = (uri or "").strip()
    if not raw:
        return raw

    # Keep non-filesystem tracking backends unchanged.
    if raw in {"databricks", "databricks-uc", "uc"}:
        return raw
    if "://" in raw:
        return raw

    # Normalize local paths (relative/absolute, including Windows paths) to file URI.
    try:
        return Path(raw).expanduser().resolve().as_uri()
    except Exception:
        return raw


def _get_mlflow_config() -> dict[str, Any]:
    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=ROOT / ".env")
    except ImportError:
        pass

    default_tracking_uri = (ROOT / "mlruns").resolve().as_uri()
    tracking_uri = _normalize_mlflow_tracking_uri(
        os.getenv("MLFLOW_TRACKING_URI", default_tracking_uri)
    )
    artifact_root = os.getenv("MLFLOW_ARTIFACT_ROOT", ".temp/mlruns")
    artifact_root_path = Path(artifact_root).expanduser()
    if not artifact_root_path.is_absolute():
        artifact_root_path = ROOT / artifact_root_path
    artifact_root_path = artifact_root_path.resolve()
    artifact_root_path.mkdir(parents=True, exist_ok=True)

    return {
        "enabled": _env_bool("MLFLOW_ENABLED", default=True),
        "tracking_uri": tracking_uri,
        "artifact_root_uri": ".temp/mlruns", 
        "artifact_root_path": str(artifact_root_path),
        "experiment_name": os.getenv("MLFLOW_EXPERIMENT_NAME", "CreditScoringTraining"),
        "run_name": os.getenv("MLFLOW_RUN_NAME", ""),
    }


def _ensure_experiment_with_artifact_root(
    client: Any,
    base_name: str,
    artifact_root_uri: str,
) -> tuple[str, str]:
    exp = client.get_experiment_by_name(base_name)
    if exp is None:
        artifact_path = Path(artifact_root_uri).expanduser().resolve()
        artifact_uri = artifact_path.as_uri() if artifact_path.is_absolute() else str(artifact_path)
        
        exp_id = client.create_experiment(
            base_name,
            artifact_location=artifact_uri,  # ← ALL RUNS sẽ dùng location này
        )
        logger.info(
            f"Created experiment '{base_name}' (id={exp_id}). "
            f"Artifact location: {artifact_uri}"
        )
        return base_name, str(exp_id)

    logger.info(
        f"Using existing experiment '{base_name}' (id={exp.experiment_id}). "
        f"Run artifacts → {exp.artifact_location}"
    )
    return str(exp.name), str(exp.experiment_id)


def _init_mlflow_state() -> dict[str, Any]:
    cfg = _get_mlflow_config()
    state: dict[str, Any] = {
        "enabled": False,
        "mlflow": None,
        "run_started": False,
        "config": cfg,
    }
    if not cfg["enabled"]:
        logger.info("MLflow tracking disabled by MLFLOW_ENABLED=false")
        return state

    try:
        import mlflow  # type: ignore
        from mlflow.tracking import MlflowClient  # type: ignore

        mlflow.set_tracking_uri(cfg["tracking_uri"])
        client = MlflowClient()
        active_experiment_name, active_experiment_id = _ensure_experiment_with_artifact_root(
            client,
            cfg["experiment_name"],
            cfg["artifact_root_uri"],
        )

        mlflow.set_experiment(cfg["experiment_name"])

        tracking_scheme = urlparse(cfg["tracking_uri"]).scheme.lower()
        if tracking_scheme in {"http", "https"}:
            logger.info(
                "Tracking server mode detected; experiment artifact location is managed by registry metadata."
            )
        state["enabled"] = True
        state["mlflow"] = mlflow
        logger.info(
            "MLflow enabled: tracking_uri=%s, experiment=%s, artifact_root=%s",
            cfg["tracking_uri"],
            cfg["experiment_name"],
            cfg["artifact_root_path"],
        )
    except Exception as e:
        logger.warning("MLflow unavailable, continue without tracking: %s", e)
    return state


def _mlflow_start_run(state: dict[str, Any], fallback_name: str) -> None:
    if not state.get("enabled") or state.get("mlflow") is None:
        return
    run_name = state["config"].get("run_name") or fallback_name
    try:
        state["mlflow"].start_run(run_name=run_name)
        state["run_started"] = True
    except Exception as e:
        logger.warning("Failed to start MLflow run, continue without MLflow: %s", e)
        state["enabled"] = False


def _mlflow_end_run(state: dict[str, Any]) -> None:
    if not state.get("enabled") or not state.get("run_started"):
        return
    try:
        state["mlflow"].end_run()
    except Exception as e:
        logger.warning("Failed to end MLflow run: %s", e)


def _safe_mlflow_log_param(state: dict[str, Any], key: str, value: Any) -> None:
    if not state.get("enabled"):
        return
    try:
        if isinstance(value, (dict, list, tuple, set)):
            value = json.dumps(value, default=str)
        state["mlflow"].log_param(key, value)
    except Exception as e:
        logger.warning("MLflow log_param failed for %s: %s", key, e)


def _safe_mlflow_log_params(state: dict[str, Any], params: dict[str, Any]) -> None:
    for key, value in params.items():
        _safe_mlflow_log_param(state, key, value)


def _safe_mlflow_log_metrics(state: dict[str, Any], metrics: dict[str, Any], prefix: str = "") -> None:
    if not state.get("enabled"):
        return
    payload: dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
            payload[f"{prefix}{key}"] = float(value)
    if not payload:
        return
    try:
        state["mlflow"].log_metrics(payload)
    except Exception as e:
        logger.warning("MLflow log_metrics failed for prefix %s: %s", prefix, e)


def _safe_mlflow_log_metric(state: dict[str, Any], key: str, value: Any) -> None:
    _safe_mlflow_log_metrics(state, {key: value})


def _safe_mlflow_log_artifact(state: dict[str, Any], path: Path, artifact_path: str | None = None) -> None:
    if not state.get("enabled") or not path.exists():
        return
    try:
        state["mlflow"].log_artifact(str(path), artifact_path=artifact_path)
    except Exception as e:
        logger.warning("MLflow log_artifact failed for %s: %s", path, e)


def _safe_mlflow_log_dir(state: dict[str, Any], directory: Path, artifact_path: str | None = None) -> None:
    if not state.get("enabled") or not directory.exists():
        return
    try:
        state["mlflow"].log_artifacts(str(directory), artifact_path=artifact_path)
    except Exception as e:
        logger.warning("MLflow log_artifacts failed for %s: %s", directory, e)


def _safe_mlflow_log_model(
    state: dict[str, Any],
    model: Any,
    artifact_path: str,
    registered_model_name: str | None = None,
) -> bool:
    if not state.get("enabled"):
        return False
    try:
        mlflow = state["mlflow"]
        try:
            import mlflow.sklearn  # type: ignore

            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path=artifact_path,
                registered_model_name=registered_model_name,
            )
            return True
        except Exception:
            pass

        mlflow.pyfunc.log_model(
            artifact_path=artifact_path,
            python_model=model,
            registered_model_name=registered_model_name,
        )
        return True
    except Exception as e:
        logger.warning("MLflow log_model failed, fallback to bundle artifact: %s", e)
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Model factory
# ─────────────────────────────────────────────────────────────────────────────

def _build_models() -> dict:
    """Return 5 untrained sklearn-compatible classifiers."""
    from sklearn.linear_model  import LogisticRegression
    from sklearn.ensemble      import RandomForestClassifier, ExtraTreesClassifier

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
            "eval_set": [(X_valid, y_valid)],
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
    except TypeError as e:
        logger.warning("fit_kwargs rejected by %s, retrying without: %s", type(model).__name__, e)
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
    fixed_params: dict | None = None,
) -> tuple[object, dict, dict]:
    """
    Light Optuna tuning for the given model name.
    If fixed_params is provided, skip the Optuna search and train once
    with those params (used when hyperparameter_tuning_pipeline.py has
    already computed the best params).
    Returns (best_model, best_params, best_metrics).
    """
    # Fast path: pre-computed params from HPO pipeline
    if fixed_params is not None:
        from sklearn.metrics import f1_score, accuracy_score, precision_score
        from sklearn.metrics import recall_score, roc_auc_score
        from sklearn.utils.class_weight import compute_sample_weight
        from src.pipelines.hyperparameter_tuning_pipeline import _build_model

        m = _build_model(name, fixed_params)
        fit_kwargs: dict = {}
        if "xgboost" in type(m).__module__:
            sw = compute_sample_weight("balanced", y_train)
            fit_kwargs = {"sample_weight": sw,
                          "eval_set": [(X_valid, y_valid)], "verbose": False}
        elif "lightgbm" in type(m).__module__:
            sw = compute_sample_weight("balanced", y_train)
            fit_kwargs = {"sample_weight": sw,
                          "eval_set": [(X_valid, y_valid)], "callbacks": []}
        m.fit(X_train, y_train, **fit_kwargs)
        y_prob = m.predict_proba(X_valid)
        y_pred = np.argmax(y_prob, axis=1)
        metrics = {
            "model_name":      name,
            "f1_macro":        round(f1_score(y_valid, y_pred, average="macro",    zero_division=0), 4),
            "f1_weighted":     round(f1_score(y_valid, y_pred, average="weighted", zero_division=0), 4),
            "accuracy":        round(accuracy_score(y_valid, y_pred), 4),
            "precision_macro": round(precision_score(y_valid, y_pred, average="macro", zero_division=0), 4),
            "recall_macro":    round(recall_score(y_valid, y_pred, average="macro",    zero_division=0), 4),
            "auc_ovr":         round(roc_auc_score(y_valid, y_prob, multi_class="ovr", average="macro"), 4),
            "best_params":     fixed_params,
        }
        logger.info(f"  Trained {name} with HPO params: f1_macro={metrics['f1_macro']:.4f}")
        return m, fixed_params, metrics
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
                        eval_set=[(X_valid, y_valid)], verbose=False)
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

    mlflow_state = _init_mlflow_state()
    fallback_run_name = f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    _mlflow_start_run(mlflow_state, fallback_run_name)
    _safe_mlflow_log_params(
        mlflow_state,
        {
            "seed": SEED,
            "train_size": TRAIN_SIZE,
            "valid_size": VALID_SIZE,
            "test_size": TEST_SIZE,
            "target_col": TARGET_COL,
        },
    )

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
        train_raw_full, train_size=TRAIN_SIZE, valid_size=VALID_SIZE, test_size=TEST_SIZE,
        random_state=SEED,
    )
    save_splits(train_raw, valid_raw, test_raw)
    _safe_mlflow_log_params(
        mlflow_state,
        {
            "n_train_rows": len(train_raw),
            "n_valid_rows": len(valid_raw),
            "n_test_rows": len(test_raw),
        },
    )

    # ── 4. PREPROCESS ─────────────────────────────────────────────────────────
    logger.info("\n== 4. PREPROCESS ======================================")
    from src.data.preprocessing import encode_target
    train_clean, valid_clean, test_clean, _ = preprocess(train_raw, valid_raw, test_raw)
    annual_income_cap = float(train_clean["Annual_Income"].max())
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
    _safe_mlflow_log_params(
        mlflow_state,
        {
            "n_features_after_selection": X_train.shape[1],
            "annual_income_cap": float(annual_income_cap),
        },
    )

    # ── 9. TRAIN 5 CANDIDATE MODELS ───────────────────────────────────────────
    logger.info("\n== 9. TRAIN 5 MODELS ==================================")
    candidate_models = _build_models()
    _safe_mlflow_log_param(mlflow_state, "candidate_models", list(candidate_models.keys()))
    raw_results:  list[dict]  = []
    fitted_models: dict[str, object] = {}

    for name, model in candidate_models.items():
        fitted, metrics = _train_evaluate(name, model, X_train, y_train, X_valid, y_valid)
        raw_results.append(metrics)
        fitted_models[name] = fitted
        _safe_mlflow_log_metrics(mlflow_state, metrics, prefix=f"raw_{name}_")

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
    _safe_mlflow_log_param(mlflow_state, "top3_models", top3_names)

    # Save 5-model comparison CSV (Power BI)
    raw_df = pd.DataFrame(raw_results)
    write_csv(raw_df, REPORT_DIR / "raw_model_comparison.csv")

    # ── 11. TUNE TOP-3 ────────────────────────────────────────────────────────
    # If hyperparameter_tuning_pipeline.py has been run, load those params
    # instead of running a short in-line Optuna study.
    logger.info("\n== 11. TUNE TOP-3 =====================================")
    _hpo_params_path = REPORT_DIR / "best_hyperparameters.json"
    _precomputed: dict = {}
    if _hpo_params_path.exists():
        import json as _json
        with open(_hpo_params_path) as _f:
            _precomputed = _json.load(_f)
        _loaded = [n for n in top3_names if n in _precomputed]
        if _loaded:
            logger.info(f"  Loaded pre-computed HPO params for: {_loaded}")

    tuned_models:   dict[str, object] = {}
    tuning_results: list[dict]        = []
    best_params_all: dict             = {}

    for name in top3_names:
        if name in _precomputed:
            # Train with pre-computed hyperparameters (skip Optuna)
            logger.info(f"  Using HPO params for {name}: {_precomputed[name]}")
            model, params, metrics = _tune_model(
                name, X_train, y_train, X_valid, y_valid,
                fixed_params=_precomputed[name],   # train once, skip search
                n_trials=1, timeout=600,
            )
        else:
            # Light inline Optuna search (fallback when HPO pipeline not run)
            logger.info(f"  Tuning: {name}")
            model, params, metrics = _tune_model(
                name, X_train, y_train, X_valid, y_valid,
                n_trials=20, timeout=120,
            )
        tuned_models[name]    = model
        tuning_results.append(metrics)
        best_params_all[name] = params
        _safe_mlflow_log_metrics(mlflow_state, metrics, prefix=f"tuned_{name}_")

    tuning_df = pd.DataFrame(tuning_results)
    write_csv(tuning_df, REPORT_DIR / "top3_tuning_results.csv")
    write_json(best_params_all, REPORT_DIR / "best_hyperparameters.json")
    _safe_mlflow_log_param(mlflow_state, "best_params_all", best_params_all)
    logger.info(f"\n{tuning_df[['model_name','f1_macro','f1_weighted','accuracy']].to_string(index=False)}")

    # ── 12. BUILD ENSEMBLE ────────────────────────────────────────────────────
    logger.info("\n== 12. ENSEMBLE =======================================")
    ensemble, ensemble_metrics = _build_ensemble(tuned_models, X_valid, y_valid)
    _safe_mlflow_log_metrics(mlflow_state, ensemble_metrics, prefix="ensemble_")

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
    _safe_mlflow_log_params(
        mlflow_state,
        {
            "final_model_name": final_model_name,
            "final_model_type": type(final_model).__name__,
            "final_model_justification": justification,
        },
    )
    _safe_mlflow_log_metrics(mlflow_state, final_metrics, prefix="final_valid_selection_")
    write_json(
        {"final_model": final_model_name, "justification": justification, **final_metrics},
        REPORT_DIR / "final_model_selection.json",
    )

    # ── 15. FULL EVALUATION ON VALID + TEST ───────────────────────────────────
    logger.info("\n== 15. FULL EVALUATION ================================")
    eval_reports: dict[str, dict] = {}
    for X, y, split_name in [(X_valid, y_valid, "valid"), (X_test, y_test, "test")]:
        y_prob_eval = final_model.predict_proba(X)
        y_pred_eval = np.argmax(y_prob_eval, axis=1)
        report      = evaluate(y, y_pred_eval, y_prob_eval, split=split_name)
        eval_reports[split_name] = report
        _safe_mlflow_log_metrics(mlflow_state, report, prefix=f"final_{split_name}_")
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
        _safe_mlflow_log_metrics(mlflow_state, cal_report, prefix="calibration_")
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
            _safe_mlflow_log_metrics(mlflow_state, summary, prefix="fairness_")
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
    sample_df["y_true"]             = [LABEL_MAP[label_idx] for label_idx in y_test[:200]]
    sample_df["y_pred"]             = [LABEL_MAP[label_idx] for label_idx in y_pred_test[:200]]
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

    _safe_mlflow_log_params(
        mlflow_state,
        {
            "metadata_model_version": metadata["model_version"],
            "metadata_final_model_name": metadata["final_model_name"],
            "metadata_top3_models": metadata["top3_models"],
            "metadata_n_train_rows": metadata["n_train_rows"],
            "metadata_n_valid_rows": metadata["n_valid_rows"],
            "metadata_n_test_rows": metadata["n_test_rows"],
        },
    )
    _safe_mlflow_log_metric(mlflow_state, "training_time_total", metadata["training_time_total"])

    # Primary artifacts: reports, predictions, drift, and final bundle.
    _safe_mlflow_log_dir(mlflow_state, REPORT_DIR, artifact_path="reports")
    sample_path = PRED_DIR / "sample_predictions.csv"
    _safe_mlflow_log_artifact(mlflow_state, sample_path, artifact_path="predictions")
    _safe_mlflow_log_dir(mlflow_state, DRIFT_DIR, artifact_path="drift_reports")
    _safe_mlflow_log_artifact(mlflow_state, bundle_path, artifact_path="models")

    registered_model_name = "credit_score_serving"
    logged_model = _safe_mlflow_log_model(
        mlflow_state,
        serving_model,
        artifact_path="serving_model",
        registered_model_name=registered_model_name,
    )
    if not logged_model:
        _safe_mlflow_log_artifact(mlflow_state, bundle_path, artifact_path="models")

    # Register in model registry
    register_model(
        model_name    = registered_model_name,
        version       = "1.0.0",
        artifact_path = bundle_path,
        metrics       = final_metrics if isinstance(final_metrics, dict) else {},
        tags          = {"pipeline": "training_pipeline_v1"},
    )
    promote_to_production(registered_model_name, "1.0.0")

    total_time = round(time.time() - t_start, 1)
    _safe_mlflow_log_metric(mlflow_state, "training_time_total", total_time)
    logger.info(f"\n== PIPELINE COMPLETE in {total_time}s ======================")
    logger.info(f"  Final model      : {registered_model_name}")
    logger.info(f"  Bundle           : {bundle_path}")
    logger.info(f"  Reports dir      : {REPORT_DIR}")
    _mlflow_end_run(mlflow_state)
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
