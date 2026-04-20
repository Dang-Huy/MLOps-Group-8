# Model Ranking

 rank          model_name  f1_macro  f1_weighted  accuracy  auc_ovr
    1            lightgbm    0.7117       0.7231    0.7198   0.8802
    2             xgboost    0.6925       0.7031    0.6996   0.8664
    3       random_forest    0.6833       0.6916    0.6892   0.8664
    4         extra_trees    0.6502       0.6592    0.6558   0.8334
    5 logistic_regression    0.5294       0.5209    0.5262   0.7303


## Selection Rationale

Top-3 selected by primary metric (macro F1): lightgbm, xgboost, random_forest

Macro F1 is preferred because this is a multiclass problem with class imbalance.