"""
src/data/ingestion.py
=====================
Load raw data from disk into DataFrames.

Input  : CSV file paths (from configs/data.yaml or direct args)
Output : pandas DataFrames — train_raw, test_raw
Role   : First step in the pipeline; touches nothing, only reads.
"""

import pandas as pd
from pathlib import Path

# Anchor to repo root (two levels up from this file: src/data/ingestion.py)
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR   = REPO_ROOT / "data" / "raw"


def load_raw(
    train_path: Path | str | None = None,
    test_path:  Path | str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load raw train and test CSV from disk.

    Paths default to <repo_root>/data/raw/*.csv so the function works
    regardless of which directory the script is launched from.

    Parameters
    ----------
    train_path : override path to train CSV
    test_path  : override path to test CSV

    Returns
    -------
    (train_raw, test_raw) — unmodified DataFrames straight from disk
    """
    train_path = Path(train_path) if train_path else RAW_DIR / "train_raw.csv"
    test_path  = Path(test_path)  if test_path  else RAW_DIR / "test_raw.csv"

    train = pd.read_csv(train_path, low_memory=False)
    test  = pd.read_csv(test_path,  low_memory=False)

    print(f"[ingestion] train_raw : {train.shape}  ← {train_path}")
    print(f"[ingestion] test_raw  : {test.shape}  ← {test_path}")
    return train, test