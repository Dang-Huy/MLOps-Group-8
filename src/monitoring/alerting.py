"""
src/monitoring/alerting.py
===========================
Alert rule evaluation for drift, performance, and latency thresholds.
"""
from __future__ import annotations

from datetime import datetime
from src.utils.logger import get_logger

logger = get_logger(__name__)

THRESHOLDS = {
    "psi_drift":            0.20,
    "f1_macro_min":         0.65,
    "missing_rate_critical":0.30,
    "p95_latency_ms":       500.0,
    "error_rate":           0.05,
}


def _alert(alert_type: str, severity: str, message: str, action: str) -> dict:
    a = {
        "type":                alert_type,
        "severity":            severity,
        "message":             message,
        "recommended_action":  action,
        "timestamp":           datetime.utcnow().isoformat(),
    }
    logger.warning(f"[ALERT][{severity}] {message}")
    return a


def check_drift_alert(psi_max: float) -> dict | None:
    if psi_max > THRESHOLDS["psi_drift"]:
        return _alert(
            "drift", "HIGH",
            f"PSI drift detected: max_psi={psi_max:.4f} > threshold {THRESHOLDS['psi_drift']}",
            "Investigate feature distributions; schedule retraining if persistent.",
        )
    return None


def check_performance_alert(f1_macro: float) -> dict | None:
    if f1_macro < THRESHOLDS["f1_macro_min"]:
        return _alert(
            "performance", "HIGH",
            f"Model F1 degraded: f1_macro={f1_macro:.4f} < {THRESHOLDS['f1_macro_min']}",
            "Run full evaluation; trigger retraining pipeline.",
        )
    return None


def check_latency_alert(p95_ms: float) -> dict | None:
    if p95_ms > THRESHOLDS["p95_latency_ms"]:
        return _alert(
            "latency", "MEDIUM",
            f"High latency: p95={p95_ms:.1f}ms > {THRESHOLDS['p95_latency_ms']}ms",
            "Profile inference; consider model compression or batching.",
        )
    return None


def check_missing_rate_alert(col: str, rate: float) -> dict | None:
    if rate > THRESHOLDS["missing_rate_critical"]:
        return _alert(
            "data_quality", "MEDIUM",
            f"High missing rate: {col} = {rate:.1%} > {THRESHOLDS['missing_rate_critical']:.0%}",
            "Check upstream data pipeline for data quality issues.",
        )
    return None


def run_all_checks(
    psi_max: float | None = None,
    f1_macro: float | None = None,
    p95_ms: float | None = None,
) -> list[dict]:
    alerts = []
    if psi_max is not None:
        alerts.extend(filter(None, [check_drift_alert(psi_max)]))
    if f1_macro is not None:
        alerts.extend(filter(None, [check_performance_alert(f1_macro)]))
    if p95_ms is not None:
        alerts.extend(filter(None, [check_latency_alert(p95_ms)]))
    return alerts
