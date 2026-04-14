# API Specification

## Base URL
`http://localhost:8000`

## Endpoints

### GET /health
Returns API health status.

**Response 200:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "1.0.0",
  "uptime_seconds": 42.5
}
```

### POST /predict
Score a single customer record.

**Request Body (all fields optional — missing values are imputed):**
```json
{
  "Age": 30,
  "Annual_Income": 50000,
  "Monthly_Inhand_Salary": 4000,
  "Num_Bank_Accounts": 3,
  "Num_Credit_Card": 3,
  "Interest_Rate": 12,
  "Num_of_Loan": 2,
  "Delay_from_due_date": 5,
  "Outstanding_Debt": 1000,
  "Credit_Utilization_Ratio": 35,
  "Credit_History_Age": 120,
  "Credit_Mix": "Good",
  "Payment_of_Min_Amount": "Yes"
}
```

**Response 200:**
```json
{
  "predicted_class": "Good",
  "predicted_label": 2,
  "probabilities": {"Poor": 0.05, "Standard": 0.15, "Good": 0.80},
  "decision": "Low Risk",
  "action": "approve",
  "confidence": 0.80,
  "model_version": "1.0.0"
}
```

### POST /predict/batch
Score multiple records.

**Request:** `{"records": [<PredictRequest>, ...]}`
**Response:** `{"predictions": [...], "n_records": N, "model_version": "1.0.0"}`

## Decision Mapping
| Credit Score | Bracket     | Action        |
|-------------|-------------|---------------|
| Poor        | High Risk   | reject        |
| Standard    | Medium Risk | manual_review |
| Good        | Low Risk    | approve       |

*Low-confidence predictions (max prob < 0.60) always map to manual_review.*
