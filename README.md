# MLOps Group 8 — Credit Score Classification

Multiclass credit score classifier (Poor / Standard / Good) with a full MLOps stack: training pipeline, batch scoring, drift monitoring, FastAPI serving, web UI, Docker packaging, CI/CD, and Prometheus/Grafana observability.

---

## Quick Start (local)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model (generates artifacts/ and data/processed/)
python -m src.pipelines.training_pipeline

# 3. Start the API + web UI
uvicorn deployment.fastapi.main:app --host 0.0.0.0 --port 8000
```

Open <http://localhost:8000/ui/> in your browser.

---

## Quick Start (Docker Compose)

```bash
# Build and start all services (API, MLflow, Prometheus, Grafana)
docker compose up --build

# Services:
#   API + Web UI  → http://localhost:8000
#   MLflow UI     → http://localhost:5000
#   Prometheus    → http://localhost:9090
#   Grafana       → http://localhost:3000  (admin / admin)
```

---

## Project Structure

```bash
├── src/
│   ├── data/           # Loading, schema, preprocessing
│   ├── features/       # Feature engineering, encoders, imputers
│   ├── models/         # Training, evaluation, serialization, registry
│   ├── pipelines/      # Training, scoring, retraining, HPO pipelines
│   ├── serving/        # Batch scoring, decision policy, latency monitor
│   ├── monitoring/     # Input/output/drift monitors
│   ├── risk/           # PSI computation
│   └── utils/          # Logger, IO helpers
├── deployment/
│   ├── fastapi/        # FastAPI app (main.py, service, schemas, web UI)
│   └── k8s/            # Kubernetes manifests
├── monitoring/
│   ├── prometheus/     # prometheus.yml scrape config
│   └── grafana/        # Datasource + dashboard provisioning
├── artifacts/
│   ├── models/         # Model bundle, registry JSON
│   └── reports/        # Evaluation, fairness, drift reports
├── data/
│   ├── raw/            # Source data
│   ├── processed/      # Train/val/test splits
│   └── reference/      # Reference distribution for drift checks
├── tests/              # unit / integration / contract tests
├── .github/workflows/  # CI/CD (lint, test, Docker build)
├── Dockerfile
└── docker-compose.yml
```

---

## Training Pipeline

```bash
# Full training (HPO → ensemble → evaluation → registry)
python -m src.pipelines.training_pipeline

# Hyperparameter tuning only
python -m src.pipelines.hyperparameter_tuning_pipeline \
    --trials 100 --timeout 3600 --models lightgbm xgboost

# Batch scoring
python -m src.pipelines.scoring_pipeline
python -m src.pipelines.scoring_pipeline --input path/to/new_data.csv --no-drift

# Drift-gated retraining
python -m src.pipelines.retraining_pipeline
python -m src.pipelines.retraining_pipeline --force
python -m src.pipelines.retraining_pipeline --new-data path/to/new_data.csv
```

---

## API Endpoints

| Method | Path | Description |
| -------- | ------ | ------------- |
| GET | `/health` | Liveness + model version |
| GET | `/model-info` | Full model metadata (JSON) |
| POST | `/predict` | Single-record prediction |
| POST | `/predict/batch` | Batch prediction |
| GET | `/metrics` | Prometheus metrics |
| GET | `/ui/` | Home page |
| GET | `/ui/predict` | Web prediction form |
| GET | `/ui/monitor` | Monitoring dashboard |
| GET | `/docs` | Swagger UI |

### Example prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Age": 34,
    "Annual_Income": 78000,
    "Monthly_Inhand_Salary": 5200,
    "Num_Bank_Accounts": 4,
    "Num_Credit_Card": 3,
    "Interest_Rate": 12,
    "Outstanding_Debt": 1200,
    "Credit_Mix": "Good",
    "Payment_of_Min_Amount": "Yes"
  }'
```

---

## Monitoring & Observability

### Prometheus metrics (scraped from `/metrics`)

| Metric | Type | Description |
| -------- | ------ | ------------- |
| `credit_score_requests_total` | Counter | Requests by endpoint + status |
| `credit_score_request_latency_ms` | Histogram | Latency in ms by endpoint |
| `credit_score_predictions_by_class_total` | Counter | Predictions by class label |
| `credit_score_model_loaded` | Gauge | 1 if model loaded, 0 otherwise |
| `credit_score_batch_size` | Histogram | Records per batch request |

### Grafana dashboard

Pre-provisioned at <http://localhost:3000> → **Credit Score API** dashboard.  
Panels: total predictions, error rate, request rate, p50/p95/p99 latency, class distribution pie.

---

## Kubernetes Deployment

```bash
# Apply manifests (requires built image pushed to your registry)
kubectl apply -f deployment/k8s/namespace.yaml
kubectl apply -f deployment/k8s/configmap.yaml
kubectl apply -f deployment/k8s/pvc.yaml
kubectl apply -f deployment/k8s/api-deployment.yaml
kubectl apply -f deployment/k8s/api-service.yaml
kubectl apply -f deployment/k8s/hpa.yaml

# Install Prometheus + Grafana via Helm
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update
helm install monitoring prometheus-community/kube-prometheus-stack \
  -n monitoring --create-namespace \
  -f deployment/k8s/helm-values.yaml
```

The HPA scales between 2–8 replicas based on CPU (70%) and memory (80%) utilisation.

---

## CI/CD

GitHub Actions workflow at `.github/workflows/ci.yml` runs on every push:

1. **Lint** — `ruff check src/ tests/`
2. **Unit tests** — `pytest tests/unit/`
3. **Integration + contract tests** — run with `continue-on-error`
4. **Docker build** — validates the `Dockerfile` builds cleanly

To enable Docker push to GHCR, uncomment the `docker-push` job and set repository secrets.

---

## Tests

```bash
pytest tests/ -v --tb=short
pytest tests/unit/          # fast, no external deps
pytest tests/integration/   # needs model bundle
pytest tests/contract/      # needs running server
```

---

## Model Results (Ensemble Soft Voting, test set — 15k samples)

Production model: `ensemble_soft_voting` (LightGBM + XGBoost + Random Forest)

| Metric | Value |
| -------- | ------- |
| F1 Macro | 0.7876 |
| Accuracy | 0.7959 |
| AUC (OVR) | 0.9154 |
| Precision Macro | 0.7786 |
| Recall Macro | 0.7987 |

---

## Documentation

| Document | Description |
| ---------- | ------------- |
| [docs/architecture.md](docs/architecture.md) | System components, data flow, model resolution chain |
| [docs/api_spec.md](docs/api_spec.md) | REST API reference with request/response schemas |
| [docs/runbook.md](docs/runbook.md) | Operations guide: startup, retraining, troubleshooting |
| [docs/git_workflow.md](docs/git_workflow.md) | Branch strategy, commit conventions, PR workflow |
