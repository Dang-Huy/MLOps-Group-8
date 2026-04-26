"""
src/models/ensemble.py
======================
Soft-voting ensemble over heterogeneous classifiers.
"""
from __future__ import annotations

from typing import Dict, Any
import numpy as np


class SoftVotingEnsemble:
    """
    Average the predicted probabilities from multiple classifiers.

    Parameters
    ----------
    models : dict
        {name: fitted_classifier} — each must implement predict_proba(X).
    """

    def __init__(self, models: Dict[str, Any]) -> None:
        self._models = models

    def predict_proba(self, X) -> np.ndarray:
        return np.mean([m.predict_proba(X) for m in self._models.values()], axis=0)

    def predict(self, X) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)

    @property
    def classes_(self) -> np.ndarray:
        first = next(iter(self._models.values()))
        return first.classes_

    def __repr__(self) -> str:
        return f"SoftVotingEnsemble(models={list(self._models.keys())})"
