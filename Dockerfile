# ── Build stage ───────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build deps once; wheel cache stays in this layer
RUN pip install --upgrade pip wheel

COPY deployment/fastapi/requirements.txt ./deployment-requirements.txt
RUN pip install --no-cache-dir -r deployment-requirements.txt

# Install remaining src-level dependencies (the ones not in deployment/requirements)
COPY requirements.txt ./
RUN pip install --no-cache-dir \
      $(grep -v -E "^(fastapi|uvicorn|pydantic|mlflow|pandas|numpy|scikit-learn|lightgbm|joblib|python-dotenv|httpx|jinja2|aiofiles|prometheus|pytest|matplotlib|seaborn|plotly|ipykernel|notebook)" requirements.txt | grep -v '^#' | grep -v '^$') || true

# ── Runtime stage ─────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# curl is needed for Docker Compose healthchecks; libgomp1 is required by LightGBM runtime
RUN apt-get update && apt-get install -y --no-install-recommends curl libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy source code
COPY src/           ./src/
COPY deployment/    ./deployment/
COPY artifacts/     ./artifacts/
COPY data/reference ./data/reference

# Environment defaults (overridable via docker run -e or docker-compose)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MLFLOW_TRACKING_URI="" \
    MLFLOW_MODEL_NAME="" \
    MLFLOW_MODEL_ALIAS="Production" \
    MODEL_PATH_FALLBACK="" \
    MLFLOW_EXPERIMENT_ID="194323661774503133"

EXPOSE 8000

CMD ["uvicorn", "deployment.fastapi.main:app", "--host", "0.0.0.0", "--port", "8000"]
