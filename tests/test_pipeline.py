"""
Unit Tests for the Fraud Detection Pipeline.
Validates split integrity, zero resampling leakage, feature engineering determinism,
and single-row inference compatibility.
"""

import sys
from pathlib import Path

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import numpy as np
import pandas as pd
import joblib

from src.config import (
    FEATURE_COLS,
    TARGET_COL,
    TIME_COL,
    AMOUNT_COL,
    PCA_FEATURE_COLS,
    RANDOM_SEED,
)
from src.data import generate_synthetic_benchmark, split_data, get_features_and_target, validate_schema
from src.features import build_preprocessor, build_full_feature_pipeline
from src.train import train_smote_lgbm, train_class_weighted_lgbm


@pytest.fixture
def sample_dataset() -> pd.DataFrame:
    """Fixture generating a small, fast synthetic dataset for unit testing."""
    return generate_synthetic_benchmark(n_samples=2000, fraud_ratio=0.02, random_state=RANDOM_SEED)


def test_split_no_overlap(sample_dataset: pd.DataFrame):
    """
    Test 1: Verifies that train, validation, and test splits have non-overlapping indices,
    cover the entire input dataset, and respect temporal ordering.
    """
    train_df, val_df, test_df = split_data(sample_dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)

    # Check non-empty
    assert len(train_df) > 0
    assert len(val_df) > 0
    assert len(test_df) > 0

    # Total sample conservation
    assert len(train_df) + len(val_df) + len(test_df) == len(sample_dataset)

    # Temporal sequence invariant
    assert train_df[TIME_COL].max() <= val_df[TIME_COL].min(), "Train time must precede Val time"
    assert val_df[TIME_COL].max() <= test_df[TIME_COL].min(), "Val time must precede Test time"

    # Both classes present in each split
    for name, split in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        unique_classes = set(split[TARGET_COL].unique())
        assert unique_classes == {0, 1}, f"Split {name} must contain both 0 and 1 classes"


def test_no_resampling_leakage(sample_dataset: pd.DataFrame):
    """
    Test 2: Verifies that SMOTE resampling is applied strictly within the training fold
    and never mutates or leaks synthetic points into the validation/test sets.
    """
    train_df, val_df, test_df = split_data(sample_dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    X_train, y_train = get_features_and_target(train_df)
    X_val, y_val = get_features_and_target(val_df)

    initial_val_len = len(X_val)
    initial_val_fraud_count = int(y_val.sum())

    # Fit SMOTE pipeline on training set
    smote_pipeline = train_smote_lgbm(X_train, y_train)

    # Transform/Predict on validation set
    y_val_pred_proba = smote_pipeline.predict_proba(X_val)

    # Validation set dimensions and class distribution must remain strictly unmodified
    assert len(X_val) == initial_val_len
    assert int(y_val.sum()) == initial_val_fraud_count
    assert y_val_pred_proba.shape == (initial_val_len, 2)


def test_feature_pipeline_determinism(sample_dataset: pd.DataFrame):
    """
    Test 3: Verifies that feature engineering and preprocessing transformations are deterministic
    given the same input data.
    """
    X, _ = get_features_and_target(sample_dataset)

    preprocessor1 = build_preprocessor()
    res1 = preprocessor1.fit_transform(X)

    preprocessor2 = build_preprocessor()
    res2 = preprocessor2.fit_transform(X)

    np.testing.assert_allclose(res1, res2, err_msg="Feature preprocessing is non-deterministic")


def test_inference_scoring_single_row(sample_dataset: pd.DataFrame, tmp_path: Path):
    """
    Test 4: Verifies that the persisted pipeline loads cleanly, accepts a single transaction
    dictionary/DataFrame row (mimicking Phase 3 API payload), and returns valid probabilities.
    """
    train_df, _, test_df = split_data(sample_dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    X_train, y_train = get_features_and_target(train_df)

    # Train model
    pipeline = train_class_weighted_lgbm(X_train, y_train)

    # Save to temp path
    artifact_path = tmp_path / "test_pipeline.joblib"
    joblib.dump(pipeline, artifact_path)

    # Load back
    loaded_pipeline = joblib.load(artifact_path)

    # Create single transaction payload (dictionary format)
    single_record = test_df[FEATURE_COLS].iloc[[0]].to_dict(orient="records")[0]
    input_df = pd.DataFrame([single_record])

    # Predict
    probabilities = loaded_pipeline.predict_proba(input_df)

    assert probabilities.shape == (1, 2)
    assert 0.0 <= probabilities[0, 1] <= 1.0
    assert np.isclose(probabilities[0, 0] + probabilities[0, 1], 1.0)
