"""src/utils/common.py -- shared utility functions."""
import time
import functools
from typing import Callable, Any
import numpy as np
import pandas as pd


def timer(func: Callable) -> Callable:
    """Decorator: print execution time of a function."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.time()
        result = func(*args, **kwargs)
        elapsed = round(time.time() - t0, 2)
        print(f"[timer] {func.__name__} completed in {elapsed}s")
        return result
    return wrapper


def set_seed(seed: int = 42) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    """Flatten nested dict: {'a': {'b': 1}} -> {'a.b': 1}"""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    return a / b if b != 0 else default


def class_distribution(y: pd.Series | np.ndarray, label_map: dict | None = None) -> dict:
    """Return count and percentage per class."""
    s = pd.Series(y)
    counts = s.value_counts().sort_index()
    pct = (counts / len(s) * 100).round(2)
    result = {}
    for cls, cnt in counts.items():
        name = label_map.get(cls, str(cls)) if label_map else str(cls)
        result[name] = {"count": int(cnt), "pct": float(pct[cls])}
    return result
