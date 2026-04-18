"""Prometheus metrics definitions for the deployment FastAPI app."""
from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram, Info

# Request counters
REQUEST_COUNT = Counter(
    "credit_score_requests_total",
    "Total prediction requests",
    ["endpoint", "status"],
)

# Latency histogram (buckets in ms)
REQUEST_LATENCY = Histogram(
    "credit_score_request_latency_ms",
    "Prediction request latency in milliseconds",
    ["endpoint"],
    buckets=[5, 10, 25, 50, 100, 250, 500, 1000, 2500],
)

# Prediction class distribution
PREDICTION_LABELS = Counter(
    "credit_score_predictions_by_class_total",
    "Predicted credit score classes",
    ["predicted_class"],
)

# Model info gauge (version label)
MODEL_INFO = Info(
    "credit_score_model",
    "Currently loaded model metadata",
)

# Batch size histogram
BATCH_SIZE = Histogram(
    "credit_score_batch_size",
    "Number of records per batch prediction request",
    buckets=[1, 2, 5, 10, 25, 50, 100, 250, 500],
)

# Active model loaded flag
MODEL_LOADED = Gauge(
    "credit_score_model_loaded",
    "1 if the model is currently loaded, 0 otherwise",
)
