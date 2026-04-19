# API Reference

Base URL: `http://localhost:8000`

---

## Endpoints

### `GET /health`

Liveness check. Returns model version and status.

**Response 200**
```json
{
  "status": "ok",
  "model_version": "ensemble_soft_voting v1.0.0",
  "model_loaded": true
}
```

---

### `GET /model-info`

Full model metadata as JSON.

**Response 200**
```json
{
  "model_name": "ensemble_soft_voting",
  "model_version": "1.0.0",
  "source_resolved_from": "json_fallback",
  "metrics_core": {
    "f1_macro": 0.7876,
    "accuracy": 0.7959,
    "auc_ovr": 0.9154
  },
  "warnings": []
}
```

---

### `POST /predict`

Single-record credit score prediction.

**Request** — all fields optional; more fields improve accuracy.

```json
{
  "Age": 34,
  "Annual_Income": 78000,
  "Monthly_Inhand_Salary": 5200,
  "Num_Bank_Accounts": 4,
  "Num_Credit_Card": 3,
  "Interest_Rate": 12,
  "Num_of_Loan": 2,
  "Delay_from_due_date": 3,
  "Num_of_Delayed_Payment": 1,
  "Changed_Credit_Limit": 8.5,
  "Num_Credit_Inquiries": 2,
  "Outstanding_Debt": 1200,
  "Credit_Utilization_Ratio": 24.3,
  "Credit_History_Age": 86,
  "Total_EMI_per_month": 350,
  "Amount_invested_monthly": 500,
  "Monthly_Balance": 1800,
  "Occupation": "Engineer",
  "Credit_Mix": "Good",
  "Payment_of_Min_Amount": "Yes",
  "Payment_Behaviour": "Low_spent_Medium_value_payments"
}
```

**Response 200**
```json
{
  "predicted_class": "Good",
  "probabilities": {
    "Poor": 0.04,
    "Standard": 0.18,
    "Good": 0.78
  },
  "confidence": 0.78,
  "decision": "Approve",
  "action": "approve",
  "model_version": "ensemble_soft_voting v1.0.0"
}
```

**Classes**

| Class | Meaning | Default action |
|-------|---------|----------------|
| Good | Low credit risk | approve |
| Standard | Medium risk | review |
| Poor | High risk | reject |

**Errors**

| Code | Meaning |
|------|---------|
| 422 | Validation error — field value out of range or wrong type |
| 503 | Model not loaded |

---

### `POST /predict/batch`

Batch prediction. Request body is an array of the same objects as `/predict`.

**Request**
```json
[
  { "Age": 34, "Annual_Income": 78000 },
  { "Age": 45, "Annual_Income": 55000, "Credit_Mix": "Bad" }
]
```

**Response 200** — array of prediction objects (same shape as single `/predict`).

---

### `GET /metrics`

Prometheus-format metrics. Scraped by Prometheus every 15s.

Key metrics:

| Metric | Type | Labels |
|--------|------|--------|
| `credit_score_requests_total` | Counter | `endpoint`, `status` |
| `credit_score_request_latency_ms` | Histogram | `endpoint` |
| `credit_score_predictions_by_class_total` | Counter | `predicted_class` |
| `credit_score_model_loaded` | Gauge | — |
| `credit_score_batch_size` | Histogram | — |

---

### Web UI Routes

| Path | Description |
|------|-------------|
| `GET /ui/` | Home — active model info, tools |
| `GET /ui/predict` | Interactive prediction form |
| `GET /ui/monitor` | Performance metrics, drift, fairness |
| `GET /docs` | Swagger UI (auto-generated) |
| `GET /redoc` | ReDoc UI (auto-generated) |

---

## Field Reference

| Field | Type | Range | Notes |
|-------|------|-------|-------|
| Age | int | 18–100 | years |
| Annual_Income | float | ≥ 0 | USD |
| Monthly_Inhand_Salary | float | ≥ 0 | USD |
| Num_Bank_Accounts | int | ≥ 0 | |
| Num_Credit_Card | int | ≥ 0 | |
| Interest_Rate | float | 0–100 | % |
| Num_of_Loan | int | ≥ 0 | |
| Delay_from_due_date | int | ≥ 0 | days |
| Num_of_Delayed_Payment | int | ≥ 0 | |
| Changed_Credit_Limit | float | any | USD delta |
| Num_Credit_Inquiries | int | ≥ 0 | |
| Outstanding_Debt | float | ≥ 0 | USD |
| Credit_Utilization_Ratio | float | 0–100 | % |
| Credit_History_Age | int | ≥ 0 | months |
| Total_EMI_per_month | float | ≥ 0 | USD |
| Amount_invested_monthly | float | ≥ 0 | USD |
| Monthly_Balance | float | ≥ 0 | USD |
| Occupation | string | see below | |
| Credit_Mix | string | Good / Standard / Bad | |
| Payment_of_Min_Amount | string | Yes / No | |
| Payment_Behaviour | string | see below | |

**Occupation values:** Scientist, Teacher, Engineer, Entrepreneur, Developer, Lawyer, Media_Manager, Doctor, Journalist, Manager, Accountant, Musician, Mechanic, Writer, Architect

**Payment_Behaviour values:** High_spent_Small_value_payments, Low_spent_Small_value_payments, High_spent_Medium_value_payments, Low_spent_Medium_value_payments, High_spent_Large_value_payments, Low_spent_Large_value_payments
