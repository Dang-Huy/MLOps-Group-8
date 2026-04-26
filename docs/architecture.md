# Architecture

## System Overview

Credit Score Classification MLOps pipeline — trains a multiclass classifier (Poor / Standard / Good), serves predictions via FastAPI, and monitors model health with Prometheus/Grafana.

```mermaid
flowchart TD
    %% --- ĐỊNH NGHĨA MÀU SẮC (STYLES) ---
    classDef data fill:#e3f2fd,stroke:#1565c0,stroke-width:2px,color:#0d47a1;
    classDef process fill:#fff3e0,stroke:#ef6c00,stroke-width:2px,color:#e65100;
    classDef model fill:#fce4ec,stroke:#c2185b,stroke-width:2px,color:#880e4f;
    classDef api fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px,color:#1b5e20;
    classDef endpoint fill:#dcedc8,stroke:#558b2f,stroke-width:2px,color:#33691e,stroke-dasharray: 4 4;
    classDef infra fill:#f3e5f5,stroke:#4a148c,stroke-width:2px,color:#4a148c;
    classDef artifact fill:#fff9c4,stroke:#fbc02d,stroke-width:3px,color:#f57f17;

    %% ==========================================
    %% KHỐI 1: TRAINING PIPELINE
    %% ==========================================
    subgraph Training [Training Pipeline offline]
        D1[📄 raw CSV]:::data --> P1[⚙️ ingest]:::process
        P1 --> P2[✅ validate]:::process
        P2 --> P3[🧹 preprocess]:::process
        P3 --> P4[🎛️ HPO]:::process
        
        P4 --> M1[🌳 LightGBM / XGBoost / RF]:::model
        M1 --> M2[🧩 Ensemble]:::model
        
        M2 --> P5[📈 evaluate]:::process
        P5 --> P6[🔖 register]:::process
        P6 --> A1[📁 artifacts/]:::data
    end

    %% Gói Artifact chuyển giao
    Bundle((📦 <b>final_model_bundle.pkl</b>)):::artifact

    %% ==========================================
    %% KHỐI 2: FASTAPI SERVING
    %% ==========================================
    subgraph Serving [FastAPI Serving deployment/fastapi]
        
        subgraph Core [Core Logic]
            C1[🚀 main.py]:::api --> C2[🛠️ ModelService]:::api
            C2 --> C3[🔍 mlflow_resolver]:::api
            C3 --> C4[🤖 loaded model]:::model
        end

        subgraph Endpoints [API Endpoints]
            E1[POST /predict]:::endpoint --> E1_1[✅ schema validation]:::process --> E1_2[🧠 inference]:::process
            E2[GET /ui/*]:::endpoint --> E2_1[🌐 Jinja2 web UI]:::api
            E3[GET /metrics]:::endpoint --> E3_1[📊 Prometheus instrumentation]:::api
        end
        
        %% Mối liên kết ẩn để canh lề đẹp hơn giữa Core và Endpoints
        C4 ~~~ E1
    end

    %% ==========================================
    %% KHỐI 3 & 4: MONITORING & TRACKING
    %% ==========================================
    subgraph MLflow [MLflow Server]
        I1[⚙️ Port: 5000]:::infra
        I2[🗄️ SQLite backend]:::infra
        I1 --- I2
    end

    subgraph Monitor [Prometheus + Grafana]
        I3[📈 Prometheus :9090<br/>scrapes /metrics]:::infra
        I4[📊 Grafana :3000]:::infra
        I3 --- I4
    end

    %% ==========================================
    %% LUỒNG KẾT NỐI CHÍNH GIỮA CÁC KHỐI
    %% ==========================================
    
    %% Training truyền file cho Serving
    A1 ==> Bundle
    Bundle ==> C3

    %% MLflow interactions (Đường đứt nét vì giao tiếp qua API)
    P6 -.->|Log params/metrics| I1
    C3 -.->|Fetch model logic| I1

    %% Prometheus scraping
    E3_1 -.->|Scraped by| I3

    %% Tùy chỉnh viền Subgraph
    style Training fill:none,stroke:#e65100,stroke-width:2px,stroke-dasharray: 5 5
    style Serving fill:none,stroke:#2e7d32,stroke-width:2px,stroke-dasharray: 5 5
    style MLflow fill:#fafafa,stroke:#4a148c,stroke-width:1px
    style Monitor fill:#fafafa,stroke:#4a148c,stroke-width:1px
    style Core fill:none,stroke:none
    style Endpoints fill:none,stroke:none
```

---

## Components

### Training Pipeline (`src/pipelines/training_pipeline.py`)

Runs end-to-end from raw CSV to registered model:

1. **Ingestion** (`src/data/ingestion.py`) — loads `data/raw/train.csv`
2. **Validation** (`src/data/validation.py`) — schema checks, missing-value stats
3. **Preprocessing** (`src/features/`) — encoding, imputation, capping, scaling
4. **HPO** (Optuna) — tunes LightGBM, XGBoost, Random Forest independently
5. **Ensemble** — soft-voting over the three tuned models (`src/models/ensemble.py`)
6. **Evaluation** — per-class F1, fairness report, PSI drift vs reference set
7. **Registration** — writes to `artifacts/models/model_registry.json` and mirrors to MLflow

### Model Registry

Two-tier:

| Tier | Location | Used when |
| ------ | ---------- | ----------- |
| MLflow | `mlruns/` (local) or `http://mlflow:5000` (Docker) | Production deployments |
| JSON fallback | `artifacts/models/model_registry.json` | Local dev, MLflow unavailable |

### Model Resolution Chain (`deployment/fastapi/mlflow_resolver.py`)

```bash
1. MLflow alias "Production" for MLFLOW_MODEL_NAME
        ↓ fail
2. MLflow stage "Production" for MLFLOW_MODEL_NAME
        ↓ fail / not found
3. JSON fallback — reads model_registry.json, picks "production" entry
```

All MLflow-unavailable warnings are filtered from the UI warning bar.

### FastAPI Application (`deployment/fastapi/`)

| File | Role |
| ------ | ------ |
| `main.py` | App factory, startup/shutdown, mounts routers |
| `service.py` | `ModelService` — loads model, runs inference, applies decision policy |
| `schemas.py` | Pydantic input/output models |
| `web.py` | Jinja2 routes (`/ui/*`), data loaders for monitor page |
| `mlflow_resolver.py` | 3-tier model resolution |
| `config.py` | `AppConfig` from environment variables |

### Monitoring

- **Prometheus** scrapes `/metrics` every 15s
- **Grafana** dashboard provisioned at startup from `monitoring/grafana/provisioning/`
- **Drift** computed via PSI against `data/reference/` — results in `artifacts/drift_reports/`
- **Fairness** report in `artifacts/reports/fairness_report.csv`

---

## Docker Topology

```bash
docker-compose.yml
├── api          (credit-score-api:latest, :8000)
│   ├── volumes: ./artifacts → /app/artifacts (ro)
│   │            ./data/reference → /app/data/reference (ro)
│   └── depends_on: mlflow
├── mlflow       (ghcr.io/mlflow/mlflow:v2.13.0, :5000)
│   ├── backend: sqlite:////mlflow/db/mlflow.db  (named volume mlflow_db)
│   └── artifacts: /mlflow/artifacts
├── prometheus   (prom/prometheus:v2.51.0, :9090)
└── grafana      (grafana/grafana:10.4.0, :3000)
```

---

## Data Flow

```bash
data/raw/train.csv
    └─ training_pipeline ──► data/processed/{train,valid,test}.parquet
                         ──► artifacts/models/final_model_bundle.pkl
                         ──► artifacts/reports/*.{json,csv}
                         ──► artifacts/drift_reports/*.csv
                         ──► mlruns/ (MLflow tracking)

POST /predict
    └─ ModelService.predict()
         ├─ PredictInput (Pydantic validation)
         ├─ feature engineering (same transformers as training)
         ├─ model.predict_proba()
         ├─ DecisionPolicy → decision + action
         └─ PredictResponse
```
