"""
src/data/split.py
=================
Reproducible 70/15/15 train/valid/test split from a single source file.

Design rules
------------
- Uses stratified sampling to preserve class balance in all three splits.
- Fit caps and encoders on train only -- no leakage.
- Saves splits to data/processed/ for reproducibility.
- Call order in pipeline: ingestion -> validation -> split -> preprocessing

Split rationale
---------------
User requirement: 70% train / 15% valid / 15% test from train_raw.csv only.
"""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

from src.data.schema import TARGET_COL

REPO_ROOT     = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = REPO_ROOT / "data" / "processed"


def split_train_valid_test(
    df: pd.DataFrame,
    train_size: float = 0.70,
    valid_size: float = 0.15,
    test_size:  float = 0.15,
    random_state: int = 42,
    stratify_col: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratified 70/15/15 split from a single DataFrame.

    Parameters
    ----------
    df           : Full labelled DataFrame (train_raw.csv)
    train_size   : Fraction for training (default 0.70)
    valid_size   : Fraction for validation (default 0.15)
    test_size    : Fraction for test (default 0.15)
    random_state : Reproducibility seed
    stratify_col : Column to stratify on; defaults to TARGET_COL if present

    Returns
    -------
    (train_df, valid_df, test_df)  -- all with reset index
    """
    assert abs(train_size + valid_size + test_size - 1.0) < 1e-6, \
        "Split fractions must sum to 1.0"

    strat_col = stratify_col or (TARGET_COL if TARGET_COL in df.columns else None)
    stratify  = df[strat_col] if strat_col else None

    # Step 1: split off test (15%) from full set
    temp, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
        shuffle=True,
    )

    # Step 2: split remaining into train (70%) and valid (15%)
    # valid_ratio within temp = valid_size / (train_size + valid_size)
    valid_ratio = valid_size / (train_size + valid_size)
    temp_stratify = temp[strat_col] if strat_col else None

    train_df, valid_df = train_test_split(
        temp,
        test_size=valid_ratio,
        random_state=random_state,
        stratify=temp_stratify,
        shuffle=True,
    )

    train_df = train_df.reset_index(drop=True)
    valid_df = valid_df.reset_index(drop=True)
    test_df  = test_df.reset_index(drop=True)

    print(f"[split] train={len(train_df)} ({len(train_df)/len(df)*100:.1f}%)"
          f"  valid={len(valid_df)} ({len(valid_df)/len(df)*100:.1f}%)"
          f"  test={len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")

    if strat_col and strat_col in train_df.columns:
        for name, split in [("train", train_df), ("valid", valid_df), ("test", test_df)]:
            dist = split[strat_col].value_counts(normalize=True).round(3).to_dict()
            print(f"  {name} class dist: {dist}")

    return train_df, valid_df, test_df


def save_splits(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df:  pd.DataFrame,
    out_dir:  Path | None = None,
) -> None:
    """Save the three splits as CSV files to data/processed/."""
    out_dir = Path(out_dir or PROCESSED_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(out_dir / "train_split.csv", index=False)
    valid_df.to_csv(out_dir / "valid_split.csv", index=False)
    test_df.to_csv( out_dir / "test_split.csv",  index=False)
    print(f"[split] Splits saved -> {out_dir}")
