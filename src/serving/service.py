"""
src/serving/service.py
=======================
Inference orchestration -- loads model bundle and runs predictions.
Singleton pattern: load once at startup, predict many times.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.serialize import ModelBundle, load_bundle
from src.serving.decision_policy import apply_decision, apply_batch_decisions
from src.utils.logger import get_logger

logger = get_logger(__name__)

REPO_ROOT           = Path(__file__).resolve().parent.parent.parent
DEFAULT_MODEL_PATH  = REPO_ROOT / "artifacts" / "models" / "final_model_bundle.pkl"


class InferenceService:
    """Singleton inference service -- load once, predict many."""

    _instance: "InferenceService | None" = None

    def __init__(self, model_path: str | Path | None = None):
        path = Path(model_path or DEFAULT_MODEL_PATH)
        self.bundle:        ModelBundle = load_bundle(path)
        self.model_version: str         = self.bundle.metadata.get("model_version", "1.0.0")
        self._start_time:   float       = time.time()
        logger.info(f"InferenceService ready -- model v{self.model_version}")

    @classmethod
    def get_instance(cls, model_path: str | Path | None = None) -> "InferenceService":
        if cls._instance is None:
            cls._instance = cls(model_path)
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        cls._instance = None

    def predict_one(
        self,
        data: dict,
        confidence_threshold: float = 0.60,
    ) -> dict:
        """Run inference on a single record dict."""
        t0  = time.time()
        df  = pd.DataFrame([data])
        X   = self.bundle.transform(df)
        proba  = self.bundle.model.predict_proba(X)[0]
        result = apply_decision(proba, confidence_threshold)
        result["model_version"] = self.model_version
        result["latency_ms"]    = round((time.time() - t0) * 1000, 2)
        return result

    def predict_batch(
        self,
        records: list[dict],
        confidence_threshold: float = 0.60,
    ) -> list[dict]:
        """Run inference on a list of record dicts."""
        df     = pd.DataFrame(records)
        X      = self.bundle.transform(df)
        probas = self.bundle.model.predict_proba(X)
        results = apply_batch_decisions(probas, confidence_threshold)
        for r in results:
            r["model_version"] = self.model_version
        return results

    @property
    def uptime_seconds(self) -> float:
        return round(time.time() - self._start_time, 2)
