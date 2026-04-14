# System Architecture

## Overview

Credit Score MLOps is a production-grade multiclass classification system for assigning credit scores (Poor / Standard / Good) to customers.

## Pipeline Architecture

```
train_raw.csv
    │
    ├─ Validation (schema, missingness, duplicates)
    │
    ├─ 70/15/15 Stratified Split
    │
    ├─ Preprocessing (PII drop, numeric cleaning, outlier capping)
    │
    ├─ Imputation (median for numeric, mode for categorical)
    │
    ├─ Encoding (ordinal for Credit_Mix/Payment_Min, OHE for Occupation/Behaviour)
    │
    ├─ Feature Engineering (10 derived ratio features)
    │
    ├─ Feature Selection (explicit domain list + variance filter)
    │
    ├─ 5 Model Training (LR, RF, ET, XGBoost, LightGBM)
    │
    ├─ Ranking (macro F1 primary metric)
    │
    ├─ Optuna Tuning (top-3 models, 20 trials each)
    │
    ├─ Ensemble (soft voting of tuned top-3)
    │
    ├─ Calibration (isotonic regression)
    │
    └─ Serialization (ModelBundle: model + imputer + encoder + selector)
```

## Serving Architecture

```
Client Request
    │
    └─ FastAPI (POST /predict)
           │
           ├─ PredictRequest schema validation
           │
           ├─ InferenceService.predict_one()
           │    ├─ ModelBundle.transform() [preprocessing pipeline]
           │    └─ model.predict_proba()
           │
           └─ DecisionPolicy (map proba → business bracket)
                   └─ PredictResponse
```

## Monitoring Architecture

- Input monitoring: schema, missing rates, range violations
- Output monitoring: prediction distribution, confidence
- Drift monitoring: PSI feature drift vs. reference
- Latency monitoring: p95, mean request time
- All outputs: Power BI-ready CSVs in artifacts/reports/ and artifacts/drift_reports/
