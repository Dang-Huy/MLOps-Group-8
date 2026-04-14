# Operational Runbook

## Starting the API

```bash
# Local dev
uvicorn src.serving.api:app --host 0.0.0.0 --port 8000 --reload

# Docker
docker-compose up -d api
```

## Running Training

```bash
python -m src.pipelines.training_pipeline
# or
make train
```

## Running Batch Scoring

```bash
python -m src.pipelines.scoring_pipeline \
  --input data/processed/test_split.csv \
  --output artifacts/predictions/scored.csv
```

## Checking Drift

```python
from src.monitoring.drift_monitor import run_drift_check
import pandas as pd
current = pd.read_csv("artifacts/predictions/latest_batch.csv")
report = run_drift_check(current, label="production_2026_04")
print(report["overall_status"])
```

## Incident: Model Drift Detected
1. Check `artifacts/drift_reports/powerbi_drift_summary.csv` for drifted features
2. Investigate upstream data pipeline for feature changes
3. If persistent: trigger `python -m src.pipelines.retraining_pipeline --trigger drift`
4. Validate new model: `python -m src.pipelines.validation_pipeline`
5. If passed: `python -m src.pipelines.deployment_pipeline`

## Incident: High Latency
1. Check p95 latency in `monitoring/latency_monitor`
2. Profile: check if data preprocessing or model inference is bottleneck
3. Options: reduce input features, switch to lighter model, increase workers

## Rollback
```bash
cp artifacts/models/serving_model_bundle.pkl artifacts/models/final_model_bundle.pkl
# or redeploy the previous Docker image tag
```
