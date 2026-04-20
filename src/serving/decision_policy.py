"""
src/serving/decision_policy.py
================================
Map model output probabilities to business decisions.
Keeps prediction logic strictly separate from business rules.
"""
from __future__ import annotations
import numpy as np

CLASS_LABELS = ["Poor", "Standard", "Good"]
LABEL_MAP    = {0: "Poor", 1: "Standard", 2: "Good"}

DECISION_POLICY: dict[str, dict] = {
    "Poor":     {"bracket": "High Risk",   "action": "reject",        "risk_level": 3},
    "Standard": {"bracket": "Medium Risk", "action": "manual_review", "risk_level": 2},
    "Good":     {"bracket": "Low Risk",    "action": "approve",       "risk_level": 1},
}


def apply_decision(
    probabilities: np.ndarray,
    confidence_threshold: float = 0.60,
) -> dict:
    """
    Convert probability array [P(Poor), P(Standard), P(Good)] to a business decision.

    Parameters
    ----------
    probabilities        : 1-D array of length 3
    confidence_threshold : if max prob < threshold, override action -> manual_review

    Returns
    -------
    dict with keys: predicted_class, predicted_label, probabilities,
                    decision, action, confidence, model_version (added by caller)
    """
    probabilities  = np.asarray(probabilities, dtype=float)
    predicted_label = int(np.argmax(probabilities))
    predicted_class = LABEL_MAP[predicted_label]
    confidence      = float(probabilities[predicted_label])

    policy = DECISION_POLICY[predicted_class]
    if confidence < confidence_threshold:
        action   = "manual_review"
        decision = f"{policy['bracket']} (low confidence)"
    else:
        action   = policy["action"]
        decision = policy["bracket"]

    return {
        "predicted_class": predicted_class,
        "predicted_label": predicted_label,
        "probabilities":   {cls: round(float(p), 4) for cls, p in zip(CLASS_LABELS, probabilities)},
        "decision":        decision,
        "action":          action,
        "confidence":      round(confidence, 4),
    }


def apply_batch_decisions(
    probabilities_matrix: np.ndarray,
    confidence_threshold: float = 0.60,
) -> list[dict]:
    """Apply decision policy to every row of a (N, 3) probability matrix."""
    return [apply_decision(row, confidence_threshold) for row in probabilities_matrix]
