# Deployment FastAPI Backend (MLflow First)

This folder contains an inference backend that resolves production model sources with strict priority:

1. MLflow Model Registry alias `Production`
2. MLflow Model Registry stage `Production`
3. JSON fallback from `artifacts/models/model_registry.json`

When MLflow source is resolved, the backend also fetches run metadata (metrics and params) from experiment `194323661774503133` using this matching rule:

- `params.metadata_model_version == model_version`
- `params.final_model_name == model_name`

## Files

- `main.py`: FastAPI app and endpoints
- `service.py`: Inference service singleton
- `schemas.py`: Pydantic v2 request/response schemas
- `mlflow_resolver.py`: MLflow-first model resolver
- `config.py`: Environment and path configuration
- `tests/test_health_and_model_info.py`: Minimal endpoint tests
- `requirements.txt`: Deployment-specific dependencies

## Environment Variables

- `MLFLOW_TRACKING_URI`: default is local `mlruns` under repository root
- `MLFLOW_MODEL_NAME`: default prefers the registered model that has Production stage metadata
- `MLFLOW_MODEL_ALIAS`: default `production`
- `MODEL_PATH_FALLBACK`: optional explicit local artifact path used in JSON fallback mode
- `MLFLOW_EXPERIMENT_ID`: default `194323661774503133`

## Run Local on Windows

From repository root:

```powershell
python -m pip install -r deployment/fastapi/requirements.txt
python -m uvicorn deployment.fastapi.main:app --host 127.0.0.1 --port 8001 --reload
```

Open Swagger docs:

- "<http://127.0.0.1:8001/docs>"

## Endpoint List

- `GET /health`
- `GET /model-info`
- `POST /predict`
- `POST /predict/batch`

## Example Requests

### Health

```powershell
Invoke-RestMethod -Method Get -Uri "http://127.0.0.1:8001/health"
```

### Model Info

```powershell
Invoke-RestMethod -Method Get -Uri "http://127.0.0.1:8001/model-info"
```

### Predict

```powershell
$body = @{
  Age = 34
  Annual_Income = 55000
  Monthly_Inhand_Salary = 4300
  Num_Bank_Accounts = 3
  Num_Credit_Card = 2
  Interest_Rate = 8
  Num_of_Loan = 1
  Delay_from_due_date = 2
  Num_of_Delayed_Payment = 1
  Num_Credit_Inquiries = 2
  Outstanding_Debt = 500
  Credit_History_Age = 84
  Total_EMI_per_month = 120
  Occupation = "Engineer"
  Credit_Mix = "Good"
  Payment_of_Min_Amount = "Yes"
  Payment_Behaviour = "High_spent_Small_value_payments"
} | ConvertTo-Json

Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8001/predict" -ContentType "application/json" -Body $body
```

### Batch Predict

```powershell
$batch = @{
  records = @(
    @{
      Age = 34
      Annual_Income = 55000
      Monthly_Inhand_Salary = 4300
      Num_Bank_Accounts = 3
      Num_Credit_Card = 2
      Interest_Rate = 8
      Num_of_Loan = 1
      Delay_from_due_date = 2
      Num_of_Delayed_Payment = 1
      Num_Credit_Inquiries = 2
      Outstanding_Debt = 500
      Credit_History_Age = 84
      Total_EMI_per_month = 120
      Occupation = "Engineer"
      Credit_Mix = "Good"
      Payment_of_Min_Amount = "Yes"
      Payment_Behaviour = "High_spent_Small_value_payments"
    },
    @{
      Age = 52
      Annual_Income = 30000
      Monthly_Inhand_Salary = 2100
      Num_Bank_Accounts = 5
      Num_Credit_Card = 4
      Interest_Rate = 17
      Num_of_Loan = 3
      Delay_from_due_date = 12
      Num_of_Delayed_Payment = 7
      Num_Credit_Inquiries = 9
      Outstanding_Debt = 3200
      Credit_History_Age = 36
      Total_EMI_per_month = 480
      Occupation = "Mechanic"
      Credit_Mix = "Bad"
      Payment_of_Min_Amount = "No"
      Payment_Behaviour = "Low_spent_Large_value_payments"
    }
  )
} | ConvertTo-Json -Depth 5

Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8001/predict/batch" -ContentType "application/json" -Body $batch
```

## Run Tests

```powershell
python -m pytest deployment/fastapi/tests -v
```
