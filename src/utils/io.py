"""src/utils/io.py -- read/write helpers for CSV, parquet, JSON, pickle."""
import json
import pickle
from pathlib import Path
import pandas as pd


def read_csv(path: str | Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False, **kwargs)


def write_csv(df: pd.DataFrame, path: str | Path, index: bool = False, **kwargs) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index, **kwargs)


def read_parquet(path: str | Path, **kwargs) -> pd.DataFrame:
    return pd.read_parquet(path, **kwargs)


def write_parquet(df: pd.DataFrame, path: str | Path, **kwargs) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, **kwargs)


def read_json(path: str | Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def write_json(data: dict | list, path: str | Path, indent: int = 2) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=indent, default=str)


def save_pickle(obj, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: str | Path):
    with open(path, "rb") as f:
        return pickle.load(f)
