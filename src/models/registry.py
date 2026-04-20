"""
src/models/registry.py
=======================
Lightweight file-based model registry.
Tracks model versions, metrics, and artifact paths without requiring MLflow.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

from src.utils.logger import get_logger

REPO_ROOT    = Path(__file__).resolve().parent.parent.parent
REGISTRY_DIR = REPO_ROOT / "artifacts" / "models"
REGISTRY_FILE = REGISTRY_DIR / "model_registry.json"
logger = get_logger(__name__)


def _env_bool(name: str, default: bool = True) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _mlflow_enabled() -> bool:
    return _env_bool("MLFLOW_ENABLED", default=True)


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


def _get_mlflow_client():
    if not _mlflow_enabled():
        return None
    try:
        import mlflow  # type: ignore
        from mlflow.tracking import MlflowClient  # type: ignore

        default_tracking_uri = (REPO_ROOT / "mlruns").resolve().as_uri()
        tracking_uri = _normalize_mlflow_tracking_uri(
            os.getenv("MLFLOW_TRACKING_URI", default_tracking_uri)
        )
        mlflow.set_tracking_uri(tracking_uri)
        return MlflowClient()
    except Exception as e:
        logger.warning("MLflow client unavailable, fallback to JSON registry only: %s", e)
        return None


def _mirror_register_to_mlflow(
    model_name: str,
    version: str,
    artifact_path: str | Path,
    metrics: dict,
    params: dict,
    tags: dict,
) -> None:
    client = _get_mlflow_client()
    if client is None:
        return
    try:
        try:
            client.create_registered_model(model_name)
        except Exception:
            # Registered model may already exist.
            pass

        source_uri = Path(artifact_path).resolve().as_uri()
        model_version = client.create_model_version(
            name=model_name,
            source=source_uri,
            run_id=None,
            tags={
                "business_version": str(version),
                "artifact_path": str(artifact_path),
                "registry_backend": "json+mlflow_mirror",
            },
            description=json.dumps(
                {
                    "metrics": metrics or {},
                    "params": params or {},
                    "tags": tags or {},
                },
                default=str,
            ),
        )
        logger.info(
            "Mirrored model to MLflow registry: name=%s, mlflow_version=%s, business_version=%s",
            model_name,
            model_version.version,
            version,
        )
    except Exception as e:
        logger.warning("Failed to mirror model registration to MLflow: %s", e)


def _mirror_promote_to_mlflow(model_name: str, version: str) -> None:
    client = _get_mlflow_client()
    if client is None:
        return
    try:
        selected_version = None
        versions = client.search_model_versions(f"name='{model_name}'")
        for mv in versions:
            mv_tags = getattr(mv, "tags", {}) or {}
            if str(mv_tags.get("business_version", "")) == str(version):
                selected_version = str(mv.version)
                break

        if selected_version is None:
            logger.warning(
                "No MLflow model version found for %s business_version=%s. JSON promotion kept.",
                model_name,
                version,
            )
            return

        client.transition_model_version_stage(
            name=model_name,
            version=selected_version,
            stage="Production",
            archive_existing_versions=True,
        )
        logger.info(
            "Mirrored promotion to MLflow registry: name=%s, mlflow_version=%s, business_version=%s",
            model_name,
            selected_version,
            version,
        )
    except Exception as e:
        logger.warning("Failed to mirror promotion to MLflow: %s", e)


def _load_registry() -> dict:
    if REGISTRY_FILE.exists():
        with open(REGISTRY_FILE, "r") as f:
            return json.load(f)
    return {"models": [], "production": None}


def _save_registry(registry: dict) -> None:
    REGISTRY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY_FILE, "w") as f:
        json.dump(registry, f, indent=2, default=str)


def register_model(
    model_name: str,
    version: str,
    artifact_path: str | Path,
    metrics: dict,
    params: dict | None = None,
    tags: dict | None = None,
) -> dict:
    """
    Register a trained model with its metrics and artifact path.

    Returns the created registry entry.
    """
    try:
        artifact_path_str = Path(artifact_path).resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        artifact_path_str = str(artifact_path)

    entry = {
        "model_name":    model_name,
        "version":       version,
        "artifact_path": artifact_path_str,
        "metrics":       metrics,
        "params":        params or {},
        "tags":          tags or {},
        "registered_at": datetime.utcnow().isoformat(),
        "status":        "staging",
    }

    registry = _load_registry()
    # Remove any previous entry with same name+version
    registry["models"] = [
        m for m in registry["models"]
        if not (m["model_name"] == model_name and m["version"] == version)
    ]
    registry["models"].append(entry)
    _save_registry(registry)
    print(f"[registry] Registered '{model_name}' v{version} -> {artifact_path}")
    _mirror_register_to_mlflow(
        model_name=model_name,
        version=version,
        artifact_path=artifact_path,
        metrics=metrics,
        params=params or {},
        tags=tags or {},
    )
    return entry


def promote_to_production(model_name: str, version: str) -> None:
    """Promote a staging model to production status."""
    registry = _load_registry()
    for m in registry["models"]:
        if m["model_name"] == model_name and m["version"] == version:
            m["status"] = "production"
            registry["production"] = {"model_name": model_name, "version": version}
            print(f"[registry] Promoted '{model_name}' v{version} to production.")
            break
    else:
        raise KeyError(f"Model '{model_name}' v{version} not found in registry.")
    _save_registry(registry)
    _mirror_promote_to_mlflow(model_name=model_name, version=version)


def get_production_model() -> dict | None:
    """Return the current production model entry, or None."""
    registry = _load_registry()
    prod_ref = registry.get("production")
    if prod_ref is None:
        return None
    for m in registry["models"]:
        if (m["model_name"] == prod_ref["model_name"] and
                m["version"] == prod_ref["version"]):
            return m
    return None


def list_models(model_name: str | None = None) -> list[dict]:
    """List all registered models, optionally filtered by name."""
    registry = _load_registry()
    models = registry["models"]
    if model_name:
        models = [m for m in models if m["model_name"] == model_name]
    return models


def get_best_model(metric: str = "f1_macro") -> dict | None:
    """Return the model with highest value for the given metric."""
    registry = _load_registry()
    candidates = [m for m in registry["models"] if metric in m.get("metrics", {})]
    if not candidates:
        return None
    return max(candidates, key=lambda m: m["metrics"][metric])
