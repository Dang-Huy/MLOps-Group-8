"""
src/pipelines/deployment_pipeline.py
======================================
Deployment orchestration -- validate then promote the model.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipelines.validation_pipeline import run_validation_pipeline
from src.utils.logger import get_logger

logger = get_logger(__name__)

SOURCE_MODEL  = ROOT / "artifacts" / "models" / "final_model_bundle.pkl"
SERVING_MODEL = ROOT / "artifacts" / "models" / "serving_model_bundle.pkl"


def run_deployment_pipeline(force: bool = False) -> bool:
    logger.info("== DEPLOYMENT PIPELINE ==")

    result = run_validation_pipeline()
    if not result.get("passed", False) and not force:
        logger.error("Deployment ABORTED -- validation gate failed.")
        return False

    if SOURCE_MODEL.exists():
        shutil.copy2(SOURCE_MODEL, SERVING_MODEL)
        logger.info(f"Model promoted: {SOURCE_MODEL.name} -> {SERVING_MODEL.name}")
    else:
        logger.error(f"Source bundle not found: {SOURCE_MODEL}")
        return False

    logger.info("Deployment pipeline complete.")
    return True


if __name__ == "__main__":
    run_deployment_pipeline()
