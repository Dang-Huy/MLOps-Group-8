"""Inference service for deployment FastAPI backend."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pandas as pd

from deployment.fastapi.config import AppConfig, load_config
from deployment.fastapi.mlflow_resolver import ResolvedModelMetadata, resolve_model
from src.models.serialize import ModelBundle, load_bundle
from src.serving.decision_policy import apply_batch_decisions, apply_decision
from src.utils.logger import get_logger

logger = get_logger(__name__)


class InferenceBackendService:
    """Singleton service that resolves, loads, and serves a production model."""

    _instance: "InferenceBackendService | None" = None

    def __init__(self, config: AppConfig | None = None):
        self.config = config or load_config()
        self.resolved: ResolvedModelMetadata = resolve_model(self.config)
        self.bundle: ModelBundle = load_bundle(Path(self.resolved.model_path))
        self.model_version: str = self.resolved.model_version
        self._start_time: float = time.time()
        logger.info(
            "Inference service ready: source=%s, model=%s, version=%s",
            self.resolved.source_resolved_from,
            self.resolved.model_name,
            self.resolved.model_version,
        )

    @classmethod
    def get_instance(cls, config: AppConfig | None = None) -> "InferenceBackendService":
        if cls._instance is None:
            cls._instance = cls(config=config)
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        cls._instance = None

    def predict_one(self, data: dict[str, Any], confidence_threshold: float = 0.60) -> dict[str, Any]:
        """Run inference for a single record and apply business decision policy."""
        df = pd.DataFrame([data])
        features = self.bundle.transform(df)
        probabilities = self.bundle.model.predict_proba(features)[0]
        result = apply_decision(probabilities, confidence_threshold=confidence_threshold)
        result["model_version"] = self.model_version
        return result

    def predict_batch(
        self,
        records: list[dict[str, Any]],
        confidence_threshold: float = 0.60,
    ) -> list[dict[str, Any]]:
        """Run inference for a batch of records and apply business decision policy."""
        df = pd.DataFrame(records)
        features = self.bundle.transform(df)
        probabilities = self.bundle.model.predict_proba(features)
        predictions = apply_batch_decisions(probabilities, confidence_threshold=confidence_threshold)
        for item in predictions:
            item["model_version"] = self.model_version
        return predictions

    def get_model_info(self) -> dict[str, Any]:
        """Return metadata describing the currently loaded model source and run."""
        return {
            "model_name": self.resolved.model_name,
            "model_version": self.resolved.model_version,
            "model_source": self.resolved.model_source,
            "run_id": self.resolved.run_id,
            "alias_or_stage": self.resolved.alias_or_stage,
            "metrics_core": self.resolved.metrics_core,
            "best_params": self.resolved.best_params,
            "params": self.resolved.params,
            "source_resolved_from": self.resolved.source_resolved_from,
            "warnings": self.resolved.warnings,
        }

    @property
    def uptime_seconds(self) -> float:
        return round(time.time() - self._start_time, 2)
