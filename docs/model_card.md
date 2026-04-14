# Model Card — Credit Score Classifier

## Model Details
- **Task**: Multiclass classification (Poor / Standard / Good)
- **Algorithm**: Soft-voting ensemble of top-3 tuned models (XGBoost, LightGBM, Random Forest)
- **Version**: 1.0.0
- **Split**: 70% train / 15% validation / 15% test (stratified)

## Intended Use
Classify customers into credit score brackets to support credit decisions.
**Not intended** for automated high-stakes lending decisions without human oversight.

## Data
- Source: train_raw.csv (100,000 rows, 28 features)
- Target distribution: Standard 53%, Poor 29%, Good 18% (imbalanced)
- Class weights applied during training to compensate for imbalance

## Performance (Validation Set ~15,000 rows)
See `artifacts/reports/final_model_comparison.csv` for detailed metrics.
Primary metric: **macro F1** (robust to class imbalance in multiclass setting).

## Features
- 17 numeric features (age, income, debt ratios, loan counts, etc.)
- 4 categorical features (occupation, credit mix, payment behaviour, payment min)
- 10 engineered ratio features (debt-to-income, EMI-to-income, etc.)

## Limitations
- Credit History Age has 20% missing rate in raw data; imputed with median
- Occupation column has placeholder values (~8% missing)
- Performance may degrade on demographics underrepresented in training data

## Fairness
Fairness diagnostics run by Occupation group. See `artifacts/reports/fairness_report.csv`.
No protected attribute (race, gender) available in the dataset.

## Calibration
Isotonic regression calibration applied. See `artifacts/reports/calibration_report.json`.
