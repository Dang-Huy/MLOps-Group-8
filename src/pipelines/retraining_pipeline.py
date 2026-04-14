"""
src/pipelines/retraining_pipeline.py
======================================
Retraining workflow -- triggered by drift detection or schedule.
"""
from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipelines.training_pipeline import run_training_pipeline
from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_retraining_pipeline(
    trigger:       str = "scheduled",
    new_data_path: str | Path | None = None,
) -> None:
    ts = datetime.utcnow().isoformat()
    logger.info(f"== RETRAINING PIPELINE ==  trigger={trigger}  time={ts}")

    if new_data_path:
        logger.info(f"New data available at: {new_data_path}")
    else:
        logger.info("Retraining on existing data/raw/train_raw.csv.")

    bundle = run_training_pipeline()
    logger.info(f"Retraining complete. New model: {bundle.metadata.get('final_model_name')}")


if __name__ == "__main__":
    run_retraining_pipeline()
