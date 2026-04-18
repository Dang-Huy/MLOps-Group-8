"""FastAPI application for local model inference deployment."""
from __future__ import annotations

import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from deployment.fastapi.config import load_config
from deployment.fastapi.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictRequest,
    PredictResponse,
)
from deployment.fastapi.service import InferenceBackendService
from src.monitoring.latency_monitor import record_latency
from src.utils.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="Credit Score Classification API (Deployment)",
    description="MLflow-first multiclass credit score inference API.",
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

_service: InferenceBackendService | None = None


@app.on_event("startup")
async def startup_event() -> None:
    global _service
    try:
        cfg = load_config()
        _service = InferenceBackendService.get_instance(config=cfg)
        logger.info("Deployment inference service started successfully.")
    except Exception as exc:
        _service = None
        logger.error("Failed to initialize deployment inference service: %s", exc)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    logger.warning("Validation error at %s: %s", request.url.path, exc.errors())
    return JSONResponse(
        status_code=422,
        content={
            "detail": "Input validation failed.",
            "errors": exc.errors(),
        },
    )


def _get_service() -> InferenceBackendService:
    if _service is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Model not loaded. MLflow Alias/Stage resolution failed and JSON fallback "
                "was not available."
            ),
        )
    return _service


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health() -> HealthResponse:
    service = _get_service()
    return HealthResponse(
        status="healthy",
        model_loaded=True,
        model_version=service.model_version,
        uptime_seconds=service.uptime_seconds,
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Model"])
def model_info() -> ModelInfoResponse:
    service = _get_service()
    return ModelInfoResponse(**service.get_model_info())


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
async def predict(request: PredictRequest) -> PredictResponse:
    service = _get_service()
    start = time.time()
    try:
        payload = request.model_dump()
        result = service.predict_one(payload)
        record_latency((time.time() - start) * 1000)
        return PredictResponse(**result)
    except HTTPException:
        record_latency((time.time() - start) * 1000, is_error=True)
        raise
    except Exception as exc:
        record_latency((time.time() - start) * 1000, is_error=True)
        logger.error("Prediction error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["Inference"])
async def predict_batch(request: BatchPredictRequest) -> BatchPredictResponse:
    service = _get_service()
    start = time.time()
    try:
        records = [row.model_dump() for row in request.records]
        if not records:
            raise HTTPException(status_code=400, detail="records must contain at least one item")

        results = service.predict_batch(records)
        predictions = [PredictResponse(**row) for row in results]
        record_latency((time.time() - start) * 1000)
        return BatchPredictResponse(
            predictions=predictions,
            n_records=len(predictions),
            model_version=service.model_version,
        )
    except HTTPException:
        record_latency((time.time() - start) * 1000, is_error=True)
        raise
    except Exception as exc:
        record_latency((time.time() - start) * 1000, is_error=True)
        logger.error("Batch prediction error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
