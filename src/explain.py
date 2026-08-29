"""
Model Explainability with SHAP (SHapley Additive exPlanations).
Generates global feature impact beeswarm plots and local transaction waterfall explanations.
"""

import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import joblib

from src.config import (
    FIGURES_DIR,
    MODEL_PIPELINE_PATH,
    MODEL_METADATA_PATH,
    FEATURE_COLS,
    TARGET_COL,
    RANDOM_SEED,
)
from src.data import load_raw, split_data, get_features_and_target

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def generate_shap_explanations(
    pipeline_path: Path = MODEL_PIPELINE_PATH,
    metadata_path: Path = MODEL_METADATA_PATH,
    sample_size: int = 1000,
) -> None:
    """
    Computes TreeSHAP explanations for the persisted final pipeline and generates:
      1. Global feature importance (beeswarm summary plot)
      2. Global feature importance bar plot
      3. Local waterfall plots for:
         - True Positive (Detected Fraud)
         - False Positive (Flagged False Alarm)
         - True Negative (Correctly Approved Genuine)
    """
    if not pipeline_path.exists():
        raise FileNotFoundError(f"Model pipeline not found at {pipeline_path}. Run training first.")

    pipeline = joblib.load(pipeline_path)
    logger.info("Loaded persisted pipeline from %s", pipeline_path)

    threshold = 0.5
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
            threshold = meta.get("chosen_decision_threshold", 0.5)

    # Load test data
    df = load_raw()
    _, _, test_df = split_data(df)
    X_test, y_test = get_features_and_target(test_df)

    # Extract preprocessor and tree classifier from pipeline
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    # Transform feature data
    X_test_trans = preprocessor.transform(X_test)
    feature_names = FEATURE_COLS

    # Create background summary for TreeExplainer
    logger.info("Initializing TreeExplainer with %d background samples...", sample_size)
    rng = np.random.RandomState(RANDOM_SEED)
    sample_indices = rng.choice(len(X_test), size=min(sample_size, len(X_test)), replace=False)
    X_sample_trans = X_test_trans[sample_indices]

    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer(X_sample_trans)

    # 1. Global Beeswarm Plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample_trans, feature_names=feature_names, show=False)
    plt.title("Global Feature Impact (SHAP Beeswarm)", fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    summary_path = FIGURES_DIR / "shap_summary.png"
    plt.savefig(summary_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved SHAP summary plot to %s", summary_path)

    # 2. Local Waterfall Plots for Specific Scenarios
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    # Find representative samples
    tp_indices = np.where((y_test.values == 1) & (y_pred == 1))[0]
    fp_indices = np.where((y_test.values == 0) & (y_pred == 1))[0]
    tn_indices = np.where((y_test.values == 0) & (y_pred == 0))[0]

    scenarios = [
        ("True Positive (High-Risk Fraud Caught)", tp_indices, FIGURES_DIR / "shap_case_tp.png"),
        ("False Positive (Legitimate Flagged as Fraud)", fp_indices, FIGURES_DIR / "shap_case_fp.png"),
        ("True Negative (Normal Transaction Approved)", tn_indices, FIGURES_DIR / "shap_case_tn.png"),
    ]

    for label, indices, out_path in scenarios:
        if len(indices) == 0:
            logger.warning("No sample found for scenario '%s'", label)
            continue

        target_idx = indices[0]
        # Calculate explanation for this single instance
        single_trans = X_test_trans[target_idx : target_idx + 1]
        single_shap = explainer(single_trans)
        single_shap.feature_names = feature_names

        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(single_shap[0], show=False)
        prob = y_prob[target_idx]
        actual = y_test.values[target_idx]
        plt.title(
            f"{label}\nPredicted P(Fraud)={prob:.4f} | Actual Class={actual} | Threshold={threshold:.3f}",
            fontsize=12,
            fontweight="bold",
            pad=15,
        )
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("Saved SHAP local waterfall plot for '%s' to %s", label, out_path)


if __name__ == "__main__":
    generate_shap_explanations()
    print("\nSHAP explainability artifacts generated successfully.")
