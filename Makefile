.PHONY: install format lint test train evaluate serve docker-build

PYTHON := python
ROOT := .

install:
	pip install -r requirements.txt

format:
	ruff format src/ tests/ pipelines/

lint:
	ruff check src/ tests/ pipelines/

test:
	pytest tests/ -v --tb=short

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v

test-smoke:
	pytest tests/smoke/ -v

train:
	$(PYTHON) -m src.pipelines.training_pipeline

evaluate:
	$(PYTHON) -m src.pipelines.validation_pipeline

score:
	$(PYTHON) -m src.pipelines.scoring_pipeline

serve:
	uvicorn src.serving.api:app --host 0.0.0.0 --port 8000 --reload

docker-build:
	docker build -t credit-score-mlops:latest .

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
