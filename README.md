# Credit Score Classification — MLOps Pipeline

## Business Problem
A global finance company wants to automatically classify customers into credit score brackets (Poor / Standard / Good) to reduce manual effort and improve lending decision consistency.

## Project Objective
Build a production-grade MLOps pipeline that:
- Trains, ranks, tunes, and ensembles 5 classification models
- Serves predictions via a FastAPI REST endpoint
- Monitors model performance and data drift for Power BI reporting
- Is reproducible, modular, tested, and deployment-ready

## Repository Structure
```
credit-score-mlops/
├── configs/           # YAML configuration files
├── data/raw/          # Original datasets
├── data/processed/    # 70/15/15 splits
├── data/reference/    # Reference distributions for drift monitoring
├── src/
│   ├── data/          # Ingestion, validation, preprocessing, split
│   ├── features/      # Feature engineering, encoding, imputation, selection
│   ├── models/        # Training, evaluation, calibration, serialization, registry
│   ├── risk/          # PSI, drift, fairness, explainability
│   ├── serving/       # FastAPI API, inference service, batch scoring, decisions
│   ├── monitoring/    # Input/output/drift/latency monitors, Power BI exports
│   ├── pipelines/     # End-to-end orchestration pipelines
│   └── utils/         # Config, logger, IO helpers
├── notebooks/         # EDA, feature analysis, model experiments, error analysis
├── tests/             # Unit, integration, contract, smoke tests
├── deployment/        # Kubernetes, CI/CD, gunicorn configs
├── monitoring/        # Prometheus, Grafana, alert rules
├── artifacts/         # Saved models, reports, predictions, drift reports
└── docs/              # Architecture, API spec, model card, risk policy, runbook
```

## Dataset
- Source: `data/raw/train_raw.csv` (100,000 rows, 28 columns)
- Target: `Credit_Score` — Poor (29%), Standard (53%), Good (18%)
- Split: 70% train / 15% validation / 15% test (stratified)

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Or use make
make install
```

## Training

```bash
# Run full 5-model pipeline: train → rank → tune top-3 → ensemble → select final
python -m src.pipelines.training_pipeline
# or
make train
```

## Evaluation Summary
After training, check:
- `artifacts/reports/model_ranking.csv` — ranked 5 models
- `artifacts/reports/top3_tuning_results.csv` — tuned model comparison
- `artifacts/reports/final_model_comparison.csv` — ensemble vs tuned
- `artifacts/reports/eval_valid_final.json` — final model on validation
- `artifacts/reports/eval_test_final.json` — final model on test set

## API Usage

```bash
# Start the API
uvicorn src.serving.api:app --host 0.0.0.0 --port 8000 --reload

# Health check
curl http://localhost:8000/health

# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"Age":30,"Annual_Income":50000,"Monthly_Inhand_Salary":4000,
       "Credit_Mix":"Good","Payment_of_Min_Amount":"Yes"}'
```

## Testing

```bash
make test           # all tests
make test-unit      # unit only
make test-smoke     # smoke only
pytest tests/       # manual
```

## Deployment

```bash
# Docker
make docker-build
make docker-up

# View API docs
open http://localhost:8000/docs
```

## Monitoring (Power BI)
After training/scoring, Power BI-ready files are exported to:
- `artifacts/reports/powerbi_model_performance.csv`
- `artifacts/reports/powerbi_prediction_distribution.csv`
- `artifacts/reports/powerbi_input_quality.csv`
- `artifacts/drift_reports/powerbi_drift_summary.csv`

Connect these as flat file data sources in Power BI.

## Assumptions
1. Only `train_raw.csv` is used — split internally into 70/15/15
2. `Type_of_Loan` dropped (multi-label free text, too sparse for reliable encoding)
3. `Credit_History_Age` parsed from string ("22 Years and 3 Months") to integer months
4. Outlier capping uses 99th percentile of train for `Annual_Income`

## Limitations
- No time-based ordering enforced — assumes records are i.i.d.
- Fairness analysis limited to Occupation (no protected attributes available)
- Calibration requires sufficient data in each probability bin

## Future Improvements
1. Add time-based cross-validation if customer month ordering is meaningful
2. Implement online learning / incremental retraining
3. Add LIME explanations for individual predictions
4. Build Power BI dashboard template file (.pbix)
5. Add Prometheus metrics endpoint for real-time latency monitoring
