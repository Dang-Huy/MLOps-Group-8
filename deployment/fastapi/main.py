"""FastAPI application for local model inference deployment."""
from __future__ import annotations

import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from deployment.fastapi.config import load_config
from deployment.fastapi.metrics import (
    BATCH_SIZE,
    MODEL_INFO,
    MODEL_LOADED,
    PREDICTION_LABELS,
    REQUEST_COUNT,
    REQUEST_LATENCY,
)
from deployment.fastapi.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictRequest,
    PredictResponse,
)
from deployment.fastapi.service import InferenceBackendService
from deployment.fastapi.web import router as web_router
from src.monitoring.latency_monitor import record_latency
from src.utils.logger import get_logger

logger = get_logger(__name__)

_FASTAPI_DIR = Path(__file__).resolve().parent
_service: InferenceBackendService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _service
    try:
        cfg = load_config()
        _service = InferenceBackendService.get_instance(config=cfg)
        MODEL_LOADED.set(1)
        MODEL_INFO.info({
            "model_name":    _service.resolved.model_name,
            "model_version": _service.resolved.model_version,
            "source":        _service.resolved.source_resolved_from,
        })
        logger.info("Deployment inference service started successfully.")
    except Exception as exc:
        _service = None
        MODEL_LOADED.set(0)
        logger.error("Failed to initialize deployment inference service: %s", exc)
    yield
    InferenceBackendService.reset()


app = FastAPI(
    title="Credit Score Classification API (Deployment)",
    description="MLflow-first multiclass credit score inference API.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(_FASTAPI_DIR / "static")), name="static")
app.include_router(web_router)


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


# ── Prometheus metrics endpoint ───────────────────────────────────────────────

@app.get("/metrics", include_in_schema=False)
def prometheus_metrics():
    return PlainTextResponse(
        content=generate_latest().decode("utf-8"),
        media_type=CONTENT_TYPE_LATEST,
    )


# ── Health ────────────────────────────────────────────────────────────────────

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


# ── Inference ─────────────────────────────────────────────────────────────────

@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
async def predict(request: PredictRequest) -> PredictResponse:
    service = _get_service()
    start = time.time()
    try:
        payload = request.model_dump()
        result = service.predict_one(payload)
        elapsed_ms = (time.time() - start) * 1000

        record_latency(elapsed_ms)
        REQUEST_COUNT.labels(endpoint="/predict", status="ok").inc()
        REQUEST_LATENCY.labels(endpoint="/predict").observe(elapsed_ms)
        PREDICTION_LABELS.labels(predicted_class=result.get("predicted_class", "unknown")).inc()

        return PredictResponse(**result)
    except HTTPException:
        record_latency((time.time() - start) * 1000, is_error=True)
        REQUEST_COUNT.labels(endpoint="/predict", status="error").inc()
        raise
    except Exception as exc:
        elapsed_ms = (time.time() - start) * 1000
        record_latency(elapsed_ms, is_error=True)
        REQUEST_COUNT.labels(endpoint="/predict", status="error").inc()
        REQUEST_LATENCY.labels(endpoint="/predict").observe(elapsed_ms)
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
        elapsed_ms = (time.time() - start) * 1000

        record_latency(elapsed_ms)
        REQUEST_COUNT.labels(endpoint="/predict/batch", status="ok").inc()
        REQUEST_LATENCY.labels(endpoint="/predict/batch").observe(elapsed_ms)
        BATCH_SIZE.observe(len(records))
        for pred in predictions:
            PREDICTION_LABELS.labels(predicted_class=pred.predicted_class).inc()

        return BatchPredictResponse(
            predictions=predictions,
            n_records=len(predictions),
            model_version=service.model_version,
        )
    except HTTPException:
        record_latency((time.time() - start) * 1000, is_error=True)
        REQUEST_COUNT.labels(endpoint="/predict/batch", status="error").inc()
        raise
    except Exception as exc:
        elapsed_ms = (time.time() - start) * 1000
        record_latency(elapsed_ms, is_error=True)
        REQUEST_COUNT.labels(endpoint="/predict/batch", status="error").inc()
        REQUEST_LATENCY.labels(endpoint="/predict/batch").observe(elapsed_ms)
        logger.error("Batch prediction error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
