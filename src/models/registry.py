"""
src/models/registry.py
=======================
Lightweight file-based model registry.
Tracks model versions, metrics, and artifact paths without requiring MLflow.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT    = Path(__file__).resolve().parent.parent.parent
REGISTRY_DIR = REPO_ROOT / "artifacts" / "models"
REGISTRY_FILE = REGISTRY_DIR / "model_registry.json"


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
    entry = {
        "model_name":    model_name,
        "version":       version,
        "artifact_path": str(artifact_path),
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
