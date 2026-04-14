"""
src/models/serialize.py
=======================
Save and load the complete model bundle atomically.
"""

import joblib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT    = Path(__file__).resolve().parent.parent.parent
ARTIFACT_DIR = REPO_ROOT / "artifacts" / "models"
BUNDLE_NAME  = "model_bundle.pkl"


@dataclass
class ModelBundle:
    model:    Any
    imputer:  Any
    encoder:  Any
    selector: Any
    metadata: dict


def save_bundle(bundle: ModelBundle, path: Path | None = None) -> Path:
    path = path or (ARTIFACT_DIR / BUNDLE_NAME)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path, compress=3)
    print(f"[serialize] Bundle saved → {path}  ({path.stat().st_size / 1e6:.1f} MB)")
    return path


def load_bundle(path: Path | None = None) -> ModelBundle:
    path = path or (ARTIFACT_DIR / BUNDLE_NAME)
    if not path.exists():
        raise FileNotFoundError(f"No bundle at {path}. Run training_pipeline.py first.")
    bundle: ModelBundle = joblib.load(path)
    print(f"[serialize] Bundle loaded ← {path}")
    return bundle