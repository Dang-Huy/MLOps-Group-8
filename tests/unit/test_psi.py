"""Unit tests for PSI computation."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import pytest
from src.risk.psi import compute_psi, psi_summary, PSI_THRESHOLDS


def test_identical_distributions_psi_near_zero():
    np.random.seed(42)
    data = np.random.normal(0, 1, 1000)
    ref = pd.DataFrame({"x": data})
    cur = pd.DataFrame({"x": data + 0.001})
    result = compute_psi(ref, cur, columns=["x"])
    assert result["psi"].iloc[0] < 0.05


def test_different_distributions_psi_high():
    np.random.seed(42)
    ref = pd.DataFrame({"x": np.random.normal(0, 1, 1000)})
    cur = pd.DataFrame({"x": np.random.normal(5, 1, 1000)})
    result = compute_psi(ref, cur, columns=["x"])
    assert result["psi"].iloc[0] > PSI_THRESHOLDS["warning"]


def test_psi_summary_fields():
    psi_df = pd.DataFrame({
        "feature": ["a", "b", "c"],
        "psi": [0.05, 0.15, 0.30],
        "status": ["stable", "warning", "drift"],
    })
    summary = psi_summary(psi_df)
    assert summary["n_features"] == 3
    assert summary["n_drift"] == 1
    assert summary["max_psi"] == 0.30


def test_psi_with_missing_values():
    ref = pd.DataFrame({"x": [1.0, 2.0, np.nan, 4.0] * 100})
    cur = pd.DataFrame({"x": [1.0, 2.0, np.nan, 4.0] * 100})
    result = compute_psi(ref, cur, columns=["x"])
    assert result.shape[0] == 1
