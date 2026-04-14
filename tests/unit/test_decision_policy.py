"""Unit tests for serving decision policy."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pytest
from src.serving.decision_policy import apply_decision, apply_batch_decisions


def test_apply_decision_poor():
    proba = np.array([0.8, 0.1, 0.1])
    result = apply_decision(proba)
    assert result["predicted_class"] == "Poor"
    assert result["predicted_label"] == 0
    assert result["action"] == "reject"
    assert result["confidence"] == pytest.approx(0.8, abs=1e-4)


def test_apply_decision_good():
    proba = np.array([0.1, 0.1, 0.8])
    result = apply_decision(proba)
    assert result["predicted_class"] == "Good"
    assert result["action"] == "approve"


def test_apply_decision_standard():
    proba = np.array([0.2, 0.6, 0.2])
    result = apply_decision(proba)
    assert result["predicted_class"] == "Standard"
    assert result["action"] == "manual_review"


def test_low_confidence_overrides_action():
    proba = np.array([0.4, 0.35, 0.25])
    result = apply_decision(proba, confidence_threshold=0.60)
    assert result["action"] == "manual_review"


def test_probabilities_sum_preserved():
    proba = np.array([0.3, 0.3, 0.4])
    result = apply_decision(proba)
    assert abs(sum(result["probabilities"].values()) - 1.0) < 0.01


def test_batch_decisions():
    probas = np.array([
        [0.8, 0.1, 0.1],
        [0.1, 0.1, 0.8],
        [0.2, 0.6, 0.2],
    ])
    results = apply_batch_decisions(probas)
    assert len(results) == 3
    assert results[0]["predicted_class"] == "Poor"
    assert results[1]["predicted_class"] == "Good"
    assert results[2]["predicted_class"] == "Standard"
