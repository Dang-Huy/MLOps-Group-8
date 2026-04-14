"""
src/pipelines/scoring_pipeline.py
===================================
Batch scoring workflow.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.serving.batch_scoring import score_file
from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_scoring_pipeline(
    input_path:  str | Path | None = None,
    output_path: str | Path | None = None,
    model_path:  str | Path | None = None,
) -> Path:
    input_path  = Path(input_path  or ROOT / "data" / "processed" / "test_split.csv")
    output_path = Path(output_path or ROOT / "artifacts" / "predictions" / "test_predictions.csv")
    logger.info(f"[scoring] Scoring: {input_path}")
    score_file(input_path, model_path=model_path, output_path=output_path)
    logger.info(f"[scoring] Done -> {output_path}")
    return output_path


if __name__ == "__main__":
    run_scoring_pipeline()
