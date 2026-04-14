# Model Ranking

 rank          model_name  f1_macro  f1_weighted  accuracy  auc_ovr
    1             xgboost    0.6870       0.7114    0.7118   0.8569
    2       random_forest    0.6833       0.6916    0.6892   0.8664
    3            lightgbm    0.6504       0.6487    0.6507   0.8721
    4         extra_trees    0.6502       0.6592    0.6558   0.8334
    5 logistic_regression    0.5294       0.5209    0.5262   0.7303


## Selection Rationale

Top-3 selected by primary metric (macro F1): xgboost, random_forest, lightgbm

Macro F1 is preferred because this is a multiclass problem with class imbalance.