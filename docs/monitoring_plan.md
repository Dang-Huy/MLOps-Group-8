# Monitoring Plan

## Overview
Monitoring outputs target **Power BI** as the primary reporting layer.
All exports are tabular CSVs/Parquet refreshed on each scoring batch.

## Power BI Export Files

### artifacts/reports/powerbi_model_performance.csv
Columns: timestamp, model_version, split, metric, value
Contains: accuracy, f1_macro, f1_weighted, precision_macro, recall_macro, auc_ovr
Refresh: After each retraining or evaluation run

### artifacts/reports/powerbi_prediction_distribution.csv
Columns: timestamp, model_version, label, credit_class, count, percentage
Contains: Per-class prediction counts and percentages
Refresh: After each batch scoring run

### artifacts/reports/powerbi_input_quality.csv
Columns: timestamp, feature, issue_type, value, threshold, alert
Contains: Missing rates, out-of-range counts, unseen category counts
Refresh: After each inference batch

### artifacts/drift_reports/powerbi_drift_summary.csv
Columns: feature, psi, status, monitoring_window, timestamp, overall_status
Contains: PSI per feature, drift classification
Refresh: Scheduled (daily or weekly)

## Recommended Power BI Data Model
1. Model Performance table (star schema center)
2. Prediction Distribution table (linked by timestamp + model_version)
3. Input Quality table (linked by timestamp)
4. Drift Summary table (linked by timestamp)

## Alert Thresholds
| Signal               | Threshold | Action              |
|---------------------|-----------|---------------------|
| PSI drift           | > 0.20    | Trigger retraining  |
| F1 macro drop       | < 0.65    | Alert + investigate |
| p95 latency         | > 500ms   | Scale or optimize   |
| Missing rate        | > 30%     | Check data pipeline |
