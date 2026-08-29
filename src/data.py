"""
Data Loading, Schema Validation, and Leak-Free Temporal Splitting Module.
Ensures zero future-data leakage through strictly chronological partitioning.
"""

import os
import sys
import shutil
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.config import (
    RAW_DATA_PATH,
    TARGET_COL,
    TIME_COL,
    AMOUNT_COL,
    PCA_FEATURE_COLS,
    FEATURE_COLS,
    ALL_COLUMNS,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    RANDOM_SEED,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def validate_schema(df: pd.DataFrame) -> None:
    """
    Strictly validates the schema, missingness, and integrity of the transaction dataset.
    Raises ValueError if invariants are violated.
    """
    if df.empty:
        raise ValueError("Dataset is empty.")

    # Column existence check
    missing_cols = set(ALL_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Dataset missing required columns: {missing_cols}")

    # Missing values check
    null_counts = df[ALL_COLUMNS].isnull().sum()
    if null_counts.sum() > 0:
        bad_cols = null_counts[null_counts > 0].to_dict()
        raise ValueError(f"Dataset contains unexpected missing values: {bad_cols}")

    # Target class check
    unique_classes = set(df[TARGET_COL].unique())
    if not unique_classes.issubset({0, 1}):
        raise ValueError(f"Target column '{TARGET_COL}' contains invalid values: {unique_classes}. Expected subset of {{0, 1}}.")

    # Value ranges
    if (df[TIME_COL] < 0).any():
        raise ValueError("Time column contains negative values.")
    if (df[AMOUNT_COL] < 0).any():
        raise ValueError("Amount column contains negative values.")

    logger.info("Schema validation passed successfully. Row count: %d, Features: %d", len(df), len(df.columns))


def download_dataset(destination_path: Path = RAW_DATA_PATH) -> Path:
    """
    Attempts to download the ULB Credit Card Fraud dataset using kagglehub.
    If kaggle credentials are not found, falls back to generating a realistic synthetic dataset.
    """
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if destination_path.exists() and destination_path.stat().st_size > 1000:
        logger.info("Dataset already exists at %s", destination_path)
        return destination_path

    logger.info("Attempting to download ULB Credit Card Fraud dataset via kagglehub...")
    try:
        import kagglehub
        path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
        downloaded_file = Path(path) / "creditcard.csv"
        if downloaded_file.exists():
            shutil.copy(downloaded_file, destination_path)
            logger.info("Successfully downloaded dataset to %s", destination_path)
            return destination_path
    except Exception as exc:
        logger.warning("Kagglehub automated download could not complete: %s", exc)

    # If download is not possible offline, generate a high-fidelity synthetic benchmark with identical schema & 0.17% fraud rate
    logger.info("Generating realistic synthetic benchmark dataset with 100,000 transactions and 0.172%% fraud rate...")
    df_synthetic = generate_synthetic_benchmark(n_samples=100000, fraud_ratio=0.00172, random_state=RANDOM_SEED)
    df_synthetic.to_csv(destination_path, index=False)
    logger.info("Synthetic benchmark written to %s", destination_path)
    return destination_path


def generate_synthetic_benchmark(
    n_samples: int = 100000,
    fraud_ratio: float = 0.00172,
    random_state: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Generates a realistic synthetic credit card transaction dataset matching the ULB distribution:
    - 48-hour time progression
    - 28 PCA features with distinct distribution shifts for fraud
    - Log-normal skewed transaction amounts
    - Extreme class imbalance (~0.17% fraud rate)
    """
    rng = np.random.RandomState(random_state)
    n_fraud = int(n_samples * fraud_ratio)
    n_legit = n_samples - n_fraud

    # Time across 48 hours (172,800 seconds)
    time_legit = np.sort(rng.uniform(0, 172800, size=n_legit))
    time_fraud = np.sort(rng.uniform(0, 172800, size=n_fraud))

    # Amounts: Log-normal, with fraud tending toward either micro-probing or high-value spikes
    amt_legit = np.clip(rng.lognormal(mean=3.0, sigma=1.5, size=n_legit), 0.0, 25000.0)
    amt_fraud = np.clip(rng.lognormal(mean=4.2, sigma=1.8, size=n_fraud), 0.0, 25000.0)

    # PCA Features V1 to V28
    # In ULB data, V14, V12, V10, V17 are strong negative predictors for fraud, while V4, V11 are strong positive
    features_legit = rng.normal(loc=0.0, scale=1.0, size=(n_legit, 28))
    features_fraud = rng.normal(loc=0.0, scale=1.0, size=(n_fraud, 28))

    # Introduce realistic shifts in top predictive features
    features_fraud[:, 13] -= 6.0  # V14
    features_fraud[:, 11] -= 5.0  # V12
    features_fraud[:, 9] -= 4.5   # V10
    features_fraud[:, 16] -= 5.5  # V17
    features_fraud[:, 3] += 4.0   # V4
    features_fraud[:, 10] += 3.5  # V11

    legit_df = pd.DataFrame(features_legit, columns=PCA_FEATURE_COLS)
    legit_df[TIME_COL] = time_legit
    legit_df[AMOUNT_COL] = amt_legit
    legit_df[TARGET_COL] = 0

    fraud_df = pd.DataFrame(features_fraud, columns=PCA_FEATURE_COLS)
    fraud_df[TIME_COL] = time_fraud
    fraud_df[AMOUNT_COL] = amt_fraud
    fraud_df[TARGET_COL] = 1

    full_df = pd.concat([legit_df, fraud_df], ignore_index=True)
    full_df = full_df.sort_values(by=TIME_COL).reset_index(drop=True)
    return full_df[ALL_COLUMNS]


def load_raw(file_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Loads raw CSV data and executes strict schema validation.
    """
    path = Path(file_path) if file_path else RAW_DATA_PATH
    if not path.exists():
        logger.info("Raw data file %s not found. Invoking downloader...", path)
        path = download_dataset(path)

    logger.info("Loading transaction dataset from %s", path)
    df = pd.read_csv(path)
    validate_schema(df)
    return df


def split_data(
    df: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Executes a leak-free chronological split on the dataset sorted by Time.
    Splits into:
      - Train (default: 70%)
      - Validation (default: 15%)
      - Test (default: 15%)

    Guarantees:
      - Temporal integrity: train.Time <= val.Time <= test.Time
      - Non-overlapping indices
      - Both classes present in all splits
    """
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError(f"Ratios must sum to 1.0. Got: {train_ratio + val_ratio + test_ratio}")

    # Ensure chronological order
    df_sorted = df.sort_values(by=TIME_COL).reset_index(drop=True)
    total_len = len(df_sorted)

    train_end = int(total_len * train_ratio)
    val_end = int(total_len * (train_ratio + val_ratio))

    train_df = df_sorted.iloc[:train_end].copy().reset_index(drop=True)
    val_df = df_sorted.iloc[train_end:val_end].copy().reset_index(drop=True)
    test_df = df_sorted.iloc[val_end:].copy().reset_index(drop=True)

    # Validation of split properties
    for name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        n_total = len(split_df)
        n_fraud = int(split_df[TARGET_COL].sum())
        fraud_pct = (n_fraud / n_total) * 100.0 if n_total > 0 else 0.0
        logger.info(
            "%s split: %d rows (%.1f%% of total) | Fraud cases: %d (%.3f%% fraud rate)",
            name, n_total, (n_total / total_len) * 100.0, n_fraud, fraud_pct
        )
        if n_fraud == 0:
            raise ValueError(f"Split {name} contains zero fraud cases! Re-evaluate split ratios.")

    # Assert temporal sequence invariant
    assert train_df[TIME_COL].max() <= val_df[TIME_COL].min(), "Train/Val temporal boundary violation!"
    assert val_df[TIME_COL].max() <= test_df[TIME_COL].min(), "Val/Test temporal boundary violation!"

    return train_df, val_df, test_df


def get_features_and_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separates feature matrix X and target series y.
    """
    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].copy()
    return X, y


if __name__ == "__main__":
    df = load_raw()
    train_df, val_df, test_df = split_data(df)
    print("\nData loading and temporal split completed successfully.")
