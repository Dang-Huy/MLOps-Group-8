"""Integration tests for the full data pipeline."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pytest

RAW_TRAIN = ROOT / "data" / "raw" / "train_raw.csv"


@pytest.mark.skipif(not RAW_TRAIN.exists(), reason="raw data not available")
def test_full_data_pipeline_produces_clean_feature_matrix():
    from src.data.ingestion import load_raw
    from src.data.validation import validate_raw
    from src.data.split import split_train_valid_test
    from src.data.preprocessing import preprocess, encode_target
    from src.features.imputers import impute_missing
    from src.features.encoders import encode_features
    from src.features.build_features import build_features
    from src.features.selectors import select_features
    from src.data.schema import TARGET_COL

    train_full, _ = load_raw()
    validate_raw(train_full, split="train")

    train_r, valid_r, test_r = split_train_valid_test(
        train_full, train_size=0.70, valid_size=0.15, test_size=0.15
    )
    train_c, valid_c, test_c, _ = preprocess(train_r, valid_r, test_r)
    test_c = encode_target(test_c)

    imputer, train_i, valid_i, test_i = impute_missing(train_c, valid_c, test_c)
    encoder, train_e, valid_e, test_e = encode_features(train_i, valid_i, test_i)

    train_f = build_features(train_e)
    selector, train_s, valid_s, test_s = select_features(
        train_f, build_features(valid_e), build_features(test_e)
    )

    X_train = train_s.drop(columns=[TARGET_COL], errors="ignore").fillna(0)
    y_train = train_s[TARGET_COL].astype(int).values

    assert X_train.shape[0] == pytest.approx(69999, abs=100)
    assert X_train.shape[1] >= 40
    assert not X_train.isna().any().any()
    assert set(np.unique(y_train)) == {0, 1, 2}
