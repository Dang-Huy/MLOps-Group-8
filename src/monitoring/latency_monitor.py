"""
src/monitoring/latency_monitor.py
==================================
Rolling-window latency tracker for API requests.
"""
from __future__ import annotations

from collections import deque
from datetime import datetime
from typing import Deque
import numpy as np
import pandas as pd


class LatencyTracker:
    """Thread-compatible rolling-window latency tracker."""

    def __init__(self, window: int = 1000):
        self._latencies: Deque[float] = deque(maxlen=window)
        self._errors:    int = 0
        self._total:     int = 0

    def record(self, latency_ms: float, is_error: bool = False) -> None:
        self._latencies.append(latency_ms)
        self._total += 1
        if is_error:
            self._errors += 1

    def summary(self) -> dict:
        arr = np.array(self._latencies)
        if len(arr) == 0:
            return {"timestamp": datetime.utcnow().isoformat(), "request_count": 0}
        return {
            "timestamp":         datetime.utcnow().isoformat(),
            "request_count":     self._total,
            "error_count":       self._errors,
            "error_rate":        round(self._errors / max(self._total, 1), 4),
            "mean_latency_ms":   round(float(arr.mean()), 2),
            "p50_latency_ms":    round(float(np.percentile(arr, 50)), 2),
            "p95_latency_ms":    round(float(np.percentile(arr, 95)), 2),
            "p99_latency_ms":    round(float(np.percentile(arr, 99)), 2),
            "max_latency_ms":    round(float(arr.max()), 2),
        }

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([self.summary()])

    def reset(self) -> None:
        self._latencies.clear()
        self._errors = 0
        self._total  = 0


# Global singleton used by the FastAPI middleware
_global_tracker = LatencyTracker(window=10_000)


def record_latency(latency_ms: float, is_error: bool = False) -> None:
    _global_tracker.record(latency_ms, is_error)


def get_latency_summary() -> dict:
    return _global_tracker.summary()


def reset_tracker() -> None:
    _global_tracker.reset()
