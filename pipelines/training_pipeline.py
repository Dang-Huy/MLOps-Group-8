# src/pipelines/training_pipeline.py  (feature selection section)

import sys
from pathlib import Path

# Ensure the project root is on sys.path so `features` package is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import joblib
from features.imputers import impute_missing
from features.encoders import encode_features
from features.build_features import build_features
from features.selectors import select_features

ROOT_DIR      = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
ARTIFACT_DIR  = ROOT_DIR / "artifacts" / "models"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

# Load processed splits
train = pd.read_csv(f"{PROCESSED_DIR}/train_processed.csv")
valid = pd.read_csv(f"{PROCESSED_DIR}/valid_processed.csv")
test  = pd.read_csv(f"{PROCESSED_DIR}/test_processed.csv")

# Step 1 — Impute
imputer, train, valid, test = impute_missing(train, valid, test)

# Step 2 — Encode
encoder, train, valid, test = encode_features(train, valid, test)

# Step 3 — Build derived features
train, valid, test = build_features(train), build_features(valid), build_features(test)

# Step 4 — Select features
selector, train, valid, test = select_features(train, valid, test)

# Inspect results
X_train = train.drop(columns=["Credit_Score"])
y_train = train["Credit_Score"]
X_valid = valid.drop(columns=["Credit_Score"])
y_valid = valid["Credit_Score"]

print(f"X_train : {X_train.shape}")
print(f"X_valid : {X_valid.shape}")
print(f"Features: {selector.feature_names_out}")

# Persist fitted transformers
# joblib.dump(imputer, f"{ARTIFACT_DIR}/imputer.pkl")
# joblib.dump(encoder, f"{ARTIFACT_DIR}/encoder.pkl")
# joblib.dump(selector, f"{ARTIFACT_DIR}/selector.pkl")