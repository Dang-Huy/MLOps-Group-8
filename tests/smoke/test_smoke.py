"""Smoke tests — minimal health checks that run without the model artifact."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import pytest


def test_imports_data():
    from src.data.ingestion import load_raw
    from src.data.schema import TARGET_COL, TARGET_ENCODING
    from src.data.preprocessing import drop_pii_columns
    assert TARGET_COL == "Credit_Score"
    assert TARGET_ENCODING["Poor"] == 0


def test_imports_features():
    from src.features.build_features import build_features, DERIVED_FEATURES
    from src.features.encoders import CategoricalEncoderPipeline
    from src.features.imputers import ImputerPipeline
    assert len(DERIVED_FEATURES) >= 5


def test_imports_models():
    from src.models.evaluate import evaluate
    from src.models.calibrate import calibration_report
    from src.models.registry import list_models
    from src.serving.decision_policy import apply_decision


def test_imports_risk():
    from src.risk.psi import compute_psi
    from src.risk.fairness import fairness_report
    from src.risk.drift import compute_drift_report


def test_imports_monitoring():
    from src.monitoring.input_monitor import monitor_input
    from src.monitoring.output_monitor import monitor_output
    from src.monitoring.alerting import run_all_checks


def test_config_loading():
    from src.utils.config import load_config
    cfg = load_config("base")
    assert "project" in cfg


def test_logger():
    from src.utils.logger import get_logger
    logger = get_logger("smoke_test")
    logger.info("smoke test OK")


def test_model_bundle_exists_or_skip():
    model_path = ROOT / "artifacts" / "models" / "final_model_bundle.pkl"
    if not model_path.exists():
        pytest.skip("Model not yet trained")
    from src.models.serialize import load_bundle
    bundle = load_bundle(model_path)
    assert hasattr(bundle, "model")
    assert hasattr(bundle, "imputer")
