"""
src/models/ensemble.py
======================
Picklable soft-voting ensemble.

Defined here (not inside training_pipeline.py) so that joblib serializes
the class as `src.models.ensemble.SoftVotingEnsemble` regardless of how the
training script is invoked, making the bundle loadable by the API.
"""
from __future__ import annotations

import numpy as np


class SoftVotingEnsemble:
    """Soft-voting ensemble over a dict of pre-fitted classifiers."""

    def __init__(self, models: dict):
        self._models = models

    # sklearn clone() compatibility
    def get_params(self, deep: bool = True) -> dict:
        return {"models": self._models}

    def set_params(self, **params) -> "SoftVotingEnsemble":
        if "models" in params:
            self._models = params["models"]
        return self

    def fit(self, X, y, **kwargs):
        """No-op: base models are already fitted."""
        return self

    def predict_proba(self, X) -> np.ndarray:
        return np.mean([m.predict_proba(X) for m in self._models.values()], axis=0)

    def predict(self, X) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)

    @property
    def classes_(self) -> np.ndarray:
        return next(iter(self._models.values())).classes_
