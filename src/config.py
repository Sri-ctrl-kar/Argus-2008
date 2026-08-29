"""
Central Configuration Module.
Defines paths, random seeds, hyperparameter defaults, and business cost parameters.
Zero magic numbers elsewhere in the codebase.
"""

import sys
from pathlib import Path

# ==========================================
# Paths
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
RAW_DATA_PATH = RAW_DATA_DIR / "creditcard.csv"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PIPELINE_PATH = MODELS_DIR / "fraud_pipeline.joblib"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_PATH = REPORTS_DIR / "metrics.json"

# Ensure directories exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# Reproducibility & Seeds
# ==========================================
RANDOM_SEED = 42

# ==========================================
# Dataset Schema
# ==========================================
TARGET_COL = "Class"
TIME_COL = "Time"
AMOUNT_COL = "Amount"
PCA_FEATURE_COLS = [f"V{i}" for i in range(1, 29)]
FEATURE_COLS = PCA_FEATURE_COLS + [TIME_COL, AMOUNT_COL]
ALL_COLUMNS = [TIME_COL] + PCA_FEATURE_COLS + [AMOUNT_COL, TARGET_COL]

# ==========================================
# Split Parameters (Chronological)
# ==========================================
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ==========================================
# Business Cost Matrix & Threshold Selection
# ==========================================
# Cost of False Negative: Average unrecovered fraudulent transaction loss
COST_FALSE_NEGATIVE = 500.0
# Cost of False Positive: Customer friction, SMS notification, and support review
COST_FALSE_POSITIVE = 15.0
# Minimum acceptable fraud catch rate (recall constraint)
TARGET_RECALL = 0.80

# ==========================================
# Model Hyperparameters
# ==========================================
LOGISTIC_REGRESSION_PARAMS = {
    "C": 1.0,
    "max_iter": 1000,
    "class_weight": "balanced",
    "random_state": RANDOM_SEED,
    "solver": "lbfgs",
}

LIGHTGBM_PARAMS = {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "max_depth": 6,
    "num_leaves": 31,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_SEED,
    "verbose": -1,
    "n_jobs": -1,
}

XGBOOST_PARAMS = {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_SEED,
    "eval_metric": "logloss",
    "n_jobs": -1,
}

SMOTE_PARAMS = {
    "sampling_strategy": 0.1,  # Upsample minority to 10% of majority in training fold
    "k_neighbors": 5,
    "random_state": RANDOM_SEED,
}
