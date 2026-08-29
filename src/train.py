"""
Model Training, Imbalance Handling, and Pipeline Serialization.
Implements Baseline (Logistic Regression), Class-Weighted LightGBM/XGBoost,
and Pipeline-Isolated SMOTE Resampling.
"""

import sys
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import lightgbm as lgb
import xgboost as xgb

from src.config import (
    LOGISTIC_REGRESSION_PARAMS,
    LIGHTGBM_PARAMS,
    XGBOOST_PARAMS,
    SMOTE_PARAMS,
    MODEL_PIPELINE_PATH,
    MODEL_METADATA_PATH,
    FEATURE_COLS,
    TARGET_COL,
    RANDOM_SEED,
)
from src.data import load_raw, split_data, get_features_and_target
from src.features import build_preprocessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def train_baseline_logistic(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Pipeline:
    """
    Step 3: Baseline model.
    Trains Logistic Regression with default/balanced settings on minimally processed features.
    """
    logger.info("Training Baseline Model: Logistic Regression...")
    preprocessor = build_preprocessor()
    model = LogisticRegression(**LOGISTIC_REGRESSION_PARAMS)

    baseline_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model),
        ]
    )
    baseline_pipeline.fit(X_train, y_train)
    return baseline_pipeline


def train_class_weighted_lgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Pipeline:
    """
    Step 4 / 5: Strategy A - Class Weighting.
    Computes scale_pos_weight = (n_neg / n_pos) strictly on training set.
    """
    n_pos = int(y_train.sum())
    n_neg = int(len(y_train) - n_pos)
    scale_pos_weight = float(n_neg / n_pos) if n_pos > 0 else 1.0
    logger.info("Training Class-Weighted LightGBM (scale_pos_weight=%.2f)...", scale_pos_weight)

    params = LIGHTGBM_PARAMS.copy()
    params["scale_pos_weight"] = scale_pos_weight

    preprocessor = build_preprocessor()
    model = lgb.LGBMClassifier(**params)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model),
        ]
    )
    pipeline.fit(X_train, y_train)
    return pipeline


def train_smote_lgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> ImbPipeline:
    """
    Step 4: Strategy B - Pipeline-Isolated Resampling (SMOTE).
    Resampling occurs exclusively within the training fold via imblearn Pipeline.
    Validation and test folds are NEVER resampled.
    """
    logger.info("Training SMOTE Resampled LightGBM via imblearn.pipeline.Pipeline...")
    preprocessor = build_preprocessor()
    smote = SMOTE(**SMOTE_PARAMS)
    model = lgb.LGBMClassifier(**LIGHTGBM_PARAMS)

    pipeline = ImbPipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("resampler", smote),
            ("classifier", model),
        ]
    )
    pipeline.fit(X_train, y_train)
    return pipeline


def train_xgboost_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Pipeline:
    """
    Tuned Class-Weighted XGBoost model.
    """
    n_pos = int(y_train.sum())
    n_neg = int(len(y_train) - n_pos)
    scale_pos_weight = float(n_neg / n_pos) if n_pos > 0 else 1.0
    logger.info("Training Class-Weighted XGBoost (scale_pos_weight=%.2f)...", scale_pos_weight)

    params = XGBOOST_PARAMS.copy()
    params["scale_pos_weight"] = scale_pos_weight

    preprocessor = build_preprocessor()
    model = xgb.XGBClassifier(**params)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model),
        ]
    )
    pipeline.fit(X_train, y_train)
    return pipeline


def save_model_artifact(
    pipeline: Any,
    metadata: Dict[str, Any],
    pipeline_path: Path = MODEL_PIPELINE_PATH,
    metadata_path: Path = MODEL_METADATA_PATH,
) -> None:
    """
    Persists the full fitted pipeline (preprocessor + model) and metadata.
    """
    pipeline_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, pipeline_path)
    logger.info("Persisted model pipeline artifact to %s", pipeline_path)

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Persisted model metadata to %s", metadata_path)


def run_training_suite() -> Dict[str, Any]:
    """
    Loads data, splits chronologically, trains baseline and candidate models,
    and returns fitted pipelines for evaluation.
    """
    df = load_raw()
    train_df, val_df, test_df = split_data(df)

    X_train, y_train = get_features_and_target(train_df)
    X_val, y_val = get_features_and_target(val_df)
    X_test, y_test = get_features_and_target(test_df)

    # 1. Baseline
    baseline_model = train_baseline_logistic(X_train, y_train)

    # 2. Imbalance Comparison
    weighted_lgbm = train_class_weighted_lgbm(X_train, y_train)
    smote_lgbm = train_smote_lgbm(X_train, y_train)

    # 3. Alternative XGBoost
    weighted_xgb = train_xgboost_model(X_train, y_train)

    return {
        "data": {
            "train": (X_train, y_train),
            "val": (X_val, y_val),
            "test": (X_test, y_test),
        },
        "models": {
            "baseline_logistic": baseline_model,
            "class_weighted_lgbm": weighted_lgbm,
            "smote_lgbm": smote_lgbm,
            "class_weighted_xgb": weighted_xgb,
        },
    }


if __name__ == "__main__":
    from src.evaluate import run_full_evaluation
    print("Starting Argus pipeline training and evaluation...")
    metrics = run_full_evaluation()
    print("\nTraining & evaluation completed successfully. Models and metrics.json updated.")

