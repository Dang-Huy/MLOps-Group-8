"""Jinja2 web UI routes for the deployment FastAPI app."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import APIRouter, Request
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates

from src.utils.logger import get_logger

logger = get_logger(__name__)

_FASTAPI_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _FASTAPI_DIR.parent.parent
_REPORTS_DIR = _REPO_ROOT / "artifacts" / "reports"
_DRIFT_DIR = _REPO_ROOT / "artifacts" / "drift_reports"

TEMPLATES = Jinja2Templates(directory=str(_FASTAPI_DIR / "templates"))
router = APIRouter(tags=["Web UI"])

# ── Display maps (single source of truth — passed to every template) ──────────

MODEL_DISPLAY: dict[str, str] = {
    "lightgbm":            "LightGBM",
    "xgboost":             "XGBoost",
    "random_forest":       "Random Forest",
    "ensemble_soft_voting": "Ensemble (Soft Voting)",
    "ensemble_weighted":   "Ensemble (Weighted)",
    "catboost":            "CatBoost",
}

SOURCE_DISPLAY: dict[str, str] = {
    "json_fallback":          "Local Registry (JSON)",
    "mlflow_alias_production": "MLflow (alias)",
    "mlflow_stage_production": "MLflow (stage)",
}

_DISPLAY_CTX = {"model_display": MODEL_DISPLAY, "source_display": SOURCE_DISPLAY}


# ── Data loaders ──────────────────────────────────────────────────────────────

def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_csv_records(path: Path) -> list[dict]:
    try:
        return pd.read_csv(path, encoding="utf-8").to_dict(orient="records")
    except Exception:
        return []


def _load_monitor_data() -> dict[str, Any]:
    eval_data = _read_json(_REPORTS_DIR / "eval_test_final.json")
    selection = _read_json(_REPORTS_DIR / "final_model_selection.json")
    ranking = _read_csv_records(_REPORTS_DIR / "model_ranking.csv")
    pred_dist = _read_csv_records(_REPORTS_DIR / "powerbi_prediction_distribution.csv")
    fairness = _read_csv_records(_REPORTS_DIR / "fairness_report.csv")
    drift = _read_csv_records(_DRIFT_DIR / "powerbi_drift_summary.csv")

    latency: dict[str, Any] = {}
    try:
        from src.monitoring.latency_monitor import get_latency_summary
        latency = get_latency_summary()
    except Exception:
        pass

    return {
        "eval": eval_data,
        "selection": selection,
        "ranking": ranking,
        "pred_dist": pred_dist,
        "fairness": fairness,
        "drift": drift,
        "latency": latency,
    }


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/", include_in_schema=False)
async def root_redirect():
    return RedirectResponse(url="/ui/", status_code=307)


_MUTED_WARNING_FRAGMENTS = (
    "FutureWarning",
    "will be deprecated",
    "filesystem tracking backend",
    "filesystem model registry",
    "model registry stages will be removed",
    # Expected fallback messages when MLflow is not configured locally
    "Could not resolve MLflow alias",
    "Failed to query MLflow",
    "No Production stage model version found",
    "Registered Model with name=",
)


def _filter_warnings(warnings: list[str]) -> list[str]:
    return [
        w for w in warnings
        if not any(frag in w for frag in _MUTED_WARNING_FRAGMENTS)
    ]


@router.get("/ui/", include_in_schema=False)
async def home_page(request: Request):
    model_info: dict[str, Any] = {}
    try:
        from deployment.fastapi.main import _service
        if _service is not None:
            model_info = dict(_service.get_model_info())
            model_info["warnings"] = _filter_warnings(model_info.get("warnings", []))
    except Exception:
        pass
    return TEMPLATES.TemplateResponse(
        request, "home.html", {"active": "home", "model_info": model_info, **_DISPLAY_CTX}
    )


@router.get("/ui/predict", include_in_schema=False)
async def predict_page(request: Request):
    return TEMPLATES.TemplateResponse(
        request, "predict.html", {"active": "predict"}
    )


@router.get("/ui/monitor", include_in_schema=False)
async def monitor_page(request: Request):
    data = _load_monitor_data()
    return TEMPLATES.TemplateResponse(
        request, "monitor.html", {"active": "monitor", **data, **_DISPLAY_CTX}
    )
