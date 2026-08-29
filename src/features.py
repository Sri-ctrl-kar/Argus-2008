"""
Feature Engineering and Preprocessing Pipeline.
Implements strictly separated fit and transform methods to eliminate data leakage.
"""

import sys
from pathlib import Path

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

from src.config import PCA_FEATURE_COLS, TIME_COL, AMOUNT_COL


class TransactionFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom transformer that extracts time-based cyclical features and handles amount transforms.
    
    Transforms:
      1. Time -> Hour of Day (0-23)
      2. Cyclical encoding: sin(2*pi*hour/24), cos(2*pi*hour/24)
      3. Amount -> Log1p amount (log(1 + amount)) for heavy-tailed distribution moderation
    """

    def __init__(self, include_cyclical_time: bool = True, log_amount: bool = True):
        self.include_cyclical_time = include_cyclical_time
        self.log_amount = log_amount
        self.feature_names_in_ = None
        self.feature_names_out_ = None

    def fit(self, X: pd.DataFrame, y=None):
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = list(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms input DataFrame into engineered feature representation.
        """
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=self.feature_names_in_)

        # 1. Cyclical Time Features (Hour of day across 24-hour cycle)
        if self.include_cyclical_time and TIME_COL in df.columns:
            # Time is in seconds from start of recording
            hours = (df[TIME_COL] / 3600.0) % 24.0
            df["time_sin_hour"] = np.sin(2.0 * np.pi * hours / 24.0)
            df["time_cos_hour"] = np.cos(2.0 * np.pi * hours / 24.0)

        # 2. Log-transformed Amount
        if self.log_amount and AMOUNT_COL in df.columns:
            df["amount_log1p"] = np.log1p(np.maximum(df[AMOUNT_COL], 0.0))

        return df


def build_preprocessor() -> ColumnTransformer:
    """
    Constructs a ColumnTransformer that:
      - Applies RobustScaler to 'Amount' (resilient to extreme financial outliers)
      - Applies StandardScaler to 'Time'
      - Passes through pre-anonymized PCA features (V1 to V28) as they are already standardized
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("amount_scaler", RobustScaler(), [AMOUNT_COL]),
            ("time_scaler", StandardScaler(), [TIME_COL]),
            ("pca_features", "passthrough", PCA_FEATURE_COLS),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor


def build_full_feature_pipeline() -> Pipeline:
    """
    Constructs an end-to-end scikit-learn feature engineering and scaling pipeline.
    Fit strictly on training data; transform applied identically during inference.
    """
    pipeline = Pipeline(
        steps=[
            ("engineer", TransactionFeatureEngineer(include_cyclical_time=False, log_amount=False)),
            ("scaler", build_preprocessor()),
        ]
    )
    return pipeline
