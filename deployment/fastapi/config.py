"""Configuration for deployment FastAPI backend."""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re


@dataclass(frozen=True)
class AppConfig:
    """Application settings loaded from environment variables."""

    repo_root: Path
    mlflow_tracking_uri: str
    mlflow_model_name: str
    mlflow_model_alias: str
    model_path_fallback: Path | None
    model_registry_path: Path
    mlflow_experiment_id: str


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_optional_path(raw_value: str | None, repo_root: Path) -> Path | None:
    if not raw_value:
        return None
    candidate = Path(raw_value).expanduser()
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    return candidate.resolve()


def _has_production_stage(models_dir: Path, model_name: str) -> bool:
    model_dir = models_dir / model_name
    if not model_dir.exists():
        return False

    for meta_path in model_dir.glob("version-*/meta.yaml"):
        try:
            content = meta_path.read_text(encoding="utf-8")
        except OSError:
            continue
        if "current_stage: Production" in content:
            return True
    return False


def _has_alias(models_dir: Path, model_name: str, alias: str) -> bool:
    model_meta = models_dir / model_name / "meta.yaml"
    if not model_meta.exists():
        return False

    try:
        content = model_meta.read_text(encoding="utf-8")
    except OSError:
        return False

    alias_key = alias.strip()
    if not alias_key:
        return False
    return re.search(rf"(?im)^\s*{re.escape(alias_key)}\s*:", content) is not None


def _default_model_name(repo_root: Path) -> str:
    models_dir = repo_root / "mlruns" / "models"
    if _has_alias(models_dir, "credit_score_serving", "production"):
        return "credit_score_serving"
    if _has_alias(models_dir, "lightgbm_serving", "production"):
        return "lightgbm_serving"
    if _has_alias(models_dir, "lightgbm", "production"):
        return "lightgbm"
    if (models_dir / "lightgbm_serving").exists():
        return "lightgbm_serving"
    if (models_dir / "lightgbm").exists():
        return "lightgbm"
    return "lightgbm"


def _normalize_tracking_uri(raw_value: str | None, repo_root: Path) -> str:
    if not raw_value:
        return (repo_root / "mlruns").resolve().as_uri()

    stripped = raw_value.strip()
    if "://" in stripped:
        return stripped

    candidate = Path(stripped).expanduser()
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    return candidate.resolve().as_uri()


def load_config() -> AppConfig:
    """Build config with safe defaults for local Windows execution."""
    repo_root = _repo_root()

    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=repo_root / ".env")
    except ImportError:
        pass

    tracking_uri = _normalize_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"), repo_root)
    model_name = os.getenv("MLFLOW_MODEL_NAME", _default_model_name(repo_root))
    model_alias = os.getenv("MLFLOW_MODEL_ALIAS", "production")
    fallback_path = _resolve_optional_path(os.getenv("MODEL_PATH_FALLBACK"), repo_root)
    model_registry_path = repo_root / "artifacts" / "models" / "model_registry.json"
    experiment_id = os.getenv("MLFLOW_EXPERIMENT_ID", "194323661774503133")

    return AppConfig(
        repo_root=repo_root,
        mlflow_tracking_uri=tracking_uri,
        mlflow_model_name=model_name,
        mlflow_model_alias=model_alias,
        model_path_fallback=fallback_path,
        model_registry_path=model_registry_path,
        mlflow_experiment_id=experiment_id,
    )
