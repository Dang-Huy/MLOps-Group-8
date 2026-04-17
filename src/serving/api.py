"""
src/serving/api.py
==================
FastAPI application exposing online inference endpoints.

Endpoints
---------
GET  /health           -- health check
POST /predict          -- single-record prediction
POST /predict/batch    -- batch prediction
"""
from __future__ import annotations

import os
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from src.serving.schemas import (
    PredictRequest, PredictResponse,
    BatchPredictRequest, BatchPredictResponse,
    HealthResponse,
)
from src.serving.service import InferenceService
from src.monitoring.latency_monitor import record_latency
from src.utils.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="Credit Score Classification API",
    description="Multiclass credit score classification: Poor / Standard / Good",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

_service: InferenceService | None = None


@app.on_event("startup")
async def startup_event() -> None:
    global _service
    model_path = os.getenv("MODEL_PATH")
    try:
        _service = InferenceService.get_instance(model_path)
        logger.info("Inference service ready.")
    except Exception as e:
        logger.error(f"Failed to load model on startup: {e}")


def _get_service() -> InferenceService:
    if _service is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Ensure MODEL_PATH points to a valid bundle.",
        )
    return _service


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health() -> HealthResponse:
    svc = _get_service()
    return HealthResponse(
        status="healthy",
        model_loaded=True,
        model_version=svc.model_version,
        uptime_seconds=svc.uptime_seconds,
    )


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
async def predict(request: PredictRequest) -> PredictResponse:
    svc = _get_service()
    t0  = time.time()
    try:
        data   = request.model_dump()
        result = svc.predict_one(data)
        record_latency((time.time() - t0) * 1000)
        return PredictResponse(**result)
    except Exception as e:
        record_latency((time.time() - t0) * 1000, is_error=True)
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["Inference"])
async def predict_batch(request: BatchPredictRequest) -> BatchPredictResponse:
    svc = _get_service()
    t0  = time.time()
    try:
        records     = [r.model_dump() for r in request.records]
        results     = svc.predict_batch(records)
        predictions = [PredictResponse(**r) for r in results]
        record_latency((time.time() - t0) * 1000)
        return BatchPredictResponse(
            predictions=predictions,
            n_records=len(predictions),
            model_version=svc.model_version,
        )
    except Exception as e:
        record_latency((time.time() - t0) * 1000, is_error=True)
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
