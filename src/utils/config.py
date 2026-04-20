"""src/utils/config.py -- load and merge YAML config files with env overrides."""
from pathlib import Path
from typing import Any
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIGS_DIR = REPO_ROOT / "configs"


def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_config(*config_names: str) -> dict:
    """Load one or more YAML configs and merge them left-to-right.

    Example: load_config("base", "train") -> base.yaml merged with train.yaml
    """
    merged: dict = {}
    for name in config_names:
        path = CONFIGS_DIR / f"{name}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"Config not found: {path}")
        with open(path, "r") as f:
            cfg = yaml.safe_load(f) or {}
        merged = _deep_merge(merged, cfg)
    return merged


def get(config: dict, *keys: str, default: Any = None) -> Any:
    """Safe nested key access: get(cfg, 'training', 'seed', default=42)."""
    val = config
    for k in keys:
        if not isinstance(val, dict):
            return default
        val = val.get(k, default)
        if val is default:
            return default
    return val
