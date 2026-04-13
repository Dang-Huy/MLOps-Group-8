"""
pipelines/training_pipeline.py
================================
End-to-end training pipeline.

Run (from repo root OR from any directory)
------------------------------------------
    python pipelines/training_pipeline.py
    python -m pipelines.training_pipeline

Output artifacts
----------------
artifacts/models/model_bundle.pkl
artifacts/models/train_metadata.json
artifacts/reports/eval_train.json
artifacts/reports/eval_valid.json
"""

import sys
from pathlib import Path

# ── path fix ───────────────────────────────────────────────────────────────
# Resolves to the repo root (parent of the pipelines/ folder) so that
# `src.*` imports work no matter which directory you launch the script from.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# ──────────────────────────────────────────────────────────────────────────

import numpy as np

from src.data.ingestion      import load_raw
from src.data.validation     import validate_raw
from src.data.preprocessing  import preprocess, split_train_valid
from src.data.schema         import TARGET_COL

from src.features.imputers       import impute_missing
from src.features.encoders       import encode_features
from src.features.build_features import build_features
from src.features.selectors      import select_features

from src.models.train     import train, save_metadata
from src.models.evaluate  import evaluate, save_report
from src.models.serialize import ModelBundle, save_bundle


def run_training_pipeline() -> ModelBundle:

    # ── 1. Ingest ──────────────────────────────────────────────────────────
    print("\n══ 1. INGEST ══════════════════════════════════════════")
    train_raw, test_raw = load_raw()
    train_raw, valid_raw = split_train_valid(train_raw)
    print(f"Split → train {train_raw.shape}  valid {valid_raw.shape}  test {test_raw.shape}")

    # ── 2. Validate ────────────────────────────────────────────────────────
    print("\n══ 2. VALIDATE ════════════════════════════════════════")
    validate_raw(train_raw, split="train")
    validate_raw(valid_raw, split="valid")
    validate_raw(test_raw,  split="test")

    # ── 3. Preprocess ──────────────────────────────────────────────────────
    print("\n══ 3. PREPROCESS ══════════════════════════════════════")
    train_clean, valid_clean, test_clean = preprocess(train_raw, valid_raw, test_raw)

    # ── 4. Impute ──────────────────────────────────────────────────────────
    print("\n══ 4. IMPUTE ══════════════════════════════════════════")
    imputer, train_imp, valid_imp, test_imp = impute_missing(
        train_clean, valid_clean, test_clean
    )

    # ── 5. Encode ──────────────────────────────────────────────────────────
    print("\n══ 5. ENCODE ══════════════════════════════════════════")
    encoder, train_enc, valid_enc, test_enc = encode_features(
        train_imp, valid_imp, test_imp
    )

    # ── 6. Build features ──────────────────────────────────────────────────
    print("\n══ 6. BUILD FEATURES ══════════════════════════════════")
    train_feat = build_features(train_enc)
    valid_feat = build_features(valid_enc)
    test_feat  = build_features(test_enc)

    # ── 7. Select features ─────────────────────────────────────────────────
    print("\n══ 7. SELECT FEATURES ═════════════════════════════════")
    selector, train_sel, valid_sel, _ = select_features(
        train_feat, valid_feat, test_feat
    )

    X_train = train_sel.drop(columns=[TARGET_COL])
    y_train = train_sel[TARGET_COL].astype(int)
    X_valid = valid_sel.drop(columns=[TARGET_COL])
    y_valid = valid_sel[TARGET_COL].astype(int)

    print(f"X_train {X_train.shape}  X_valid {X_valid.shape}")

    # ── 8. Train ───────────────────────────────────────────────────────────
    print("\n══ 8. TRAIN ═══════════════════════════════════════════")
    model, metadata = train(X_train, y_train, X_valid, y_valid)
    metadata["feature_names"] = selector.feature_names_out
    save_metadata(metadata)

    # ── 9. Evaluate ────────────────────────────────────────────────────────
    print("\n══ 9. EVALUATE ════════════════════════════════════════")
    for X, y, split in [
        (X_train, y_train, "train"),
        (X_valid, y_valid, "valid"),
    ]:
        y_prob = model.predict_proba(X)
        y_pred = np.argmax(y_prob, axis=1)
        report = evaluate(y, y_pred, y_prob, split=split)
        save_report(report)

    # ── 10. Serialize ──────────────────────────────────────────────────────
    print("\n══ 10. SERIALIZE ══════════════════════════════════════")
    bundle = ModelBundle(
        model    = model,
        imputer  = imputer,
        encoder  = encoder,
        selector = selector,
        metadata = metadata,
    )
    save_bundle(bundle)

    print("\n══ PIPELINE COMPLETE ══════════════════════════════════")
    print("  artifacts/models/model_bundle.pkl")
    print("  artifacts/models/train_metadata.json")
    print("  artifacts/reports/eval_train.json")
    print("  artifacts/reports/eval_valid.json\n")

    return bundle


if __name__ == "__main__":
    run_training_pipeline()