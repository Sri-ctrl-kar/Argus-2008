"""
Comprehensive Evaluation, Threshold Optimization, and Metrics Logging Module.
Computes PR-AUC, ROC-AUC, Cost-Optimal Thresholds, Calibration Curves, and Confusion Matrices.
"""

import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, List

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    roc_curve,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.calibration import calibration_curve

from src.config import (
    FIGURES_DIR,
    METRICS_PATH,
    MODEL_PIPELINE_PATH,
    MODEL_METADATA_PATH,
    COST_FALSE_NEGATIVE,
    COST_FALSE_POSITIVE,
    TARGET_RECALL,
    FEATURE_COLS,
    RANDOM_SEED,
)
from src.data import load_raw, split_data, get_features_and_target
from src.train import run_training_suite, save_model_artifact

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def evaluate_probabilities(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    cost_fn: float = COST_FALSE_NEGATIVE,
    cost_fp: float = COST_FALSE_POSITIVE,
) -> Dict[str, Any]:
    """
    Computes full spectrum of performance metrics for binary classification.
    """
    y_pred = (y_prob >= threshold).astype(int)

    pr_auc = float(average_precision_score(y_true, y_prob))
    roc_auc = float(roc_auc_score(y_true, y_prob))

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    total_financial_loss = float(fn * cost_fn + fp * cost_fp)

    return {
        "threshold": float(threshold),
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
        "financial_cost": {
            "cost_false_negative_unit": cost_fn,
            "cost_false_positive_unit": cost_fp,
            "total_estimated_loss_usd": total_financial_loss,
        },
    }


def find_optimal_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cost_fn: float = COST_FALSE_NEGATIVE,
    cost_fp: float = COST_FALSE_POSITIVE,
    target_recall: float = TARGET_RECALL,
) -> Dict[str, Any]:
    """
    Finds:
      1. Cost-optimal threshold minimizing financial loss
      2. Recall-constrained threshold satisfying target recall (e.g. >= 80%)
      3. Max F1 threshold
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)

    best_cost = float("inf")
    best_cost_thresh = 0.5

    recall_constrained_thresh = None
    recall_constrained_precision = 0.0

    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_f1_idx = int(np.argmax(f1_scores))
    best_f1_thresh = float(thresholds[best_f1_idx]) if best_f1_idx < len(thresholds) else 0.5

    # Grid search across candidate thresholds
    candidate_thresholds = np.linspace(0.01, 0.99, 500)
    for t in candidate_thresholds:
        y_pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        cost = fn * cost_fn + fp * cost_fp
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        if cost < best_cost:
            best_cost = cost
            best_cost_thresh = float(t)

        # Track threshold meeting target recall with highest precision
        if rec >= target_recall and prec >= recall_constrained_precision:
            recall_constrained_thresh = float(t)
            recall_constrained_precision = prec

    if recall_constrained_thresh is None:
        recall_constrained_thresh = best_cost_thresh

    return {
        "cost_optimal_threshold": best_cost_thresh,
        "recall_constrained_threshold": recall_constrained_thresh,
        "max_f1_threshold": best_f1_thresh,
    }


def plot_precision_recall_curves(
    models_dict: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_path: Path = FIGURES_DIR / "pr_curve.png",
) -> None:
    """
    Generates high-resolution Precision-Recall curves comparing all models.
    """
    plt.figure(figsize=(9, 6))
    sns.set_theme(style="whitegrid")

    for name, model in models_dict.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        ap = average_precision_score(y_test, y_prob)
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        plt.plot(rec, prec, label=f"{name} (PR-AUC = {ap:.4f})", linewidth=2.0)

    # Baseline no-skill line (proportion of positive class)
    no_skill = y_test.sum() / len(y_test)
    plt.plot([0, 1], [no_skill, no_skill], linestyle="--", color="gray", label=f"No-Skill Baseline ({no_skill:.4f})")

    plt.xlabel("Recall (Fraud Detection Rate)", fontsize=12)
    plt.ylabel("Precision (Positive Predictive Value)", fontsize=12)
    plt.title("Precision-Recall Curves — Out-of-Time Test Set", fontsize=14, fontweight="bold")
    plt.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info("Saved PR curve to %s", output_path)


def plot_roc_curves(
    models_dict: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_path: Path = FIGURES_DIR / "roc_curve.png",
) -> None:
    """
    Generates ROC curves comparing all models.
    """
    plt.figure(figsize=(9, 6))
    sns.set_theme(style="whitegrid")

    for name, model in models_dict.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.plot(fpr, tpr, label=f"{name} (ROC-AUC = {auc:.4f})", linewidth=2.0)

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Chance (0.50)")
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate (Recall)", fontsize=12)
    plt.title("ROC Curves (Note: Suppresses True Negative Imbalance)", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info("Saved ROC curve to %s", output_path)


def plot_calibration_curves(
    models_dict: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_path: Path = FIGURES_DIR / "calibration_curve.png",
) -> None:
    """
    Generates Probability Calibration curves.
    """
    plt.figure(figsize=(9, 6))
    sns.set_theme(style="whitegrid")

    for name, model in models_dict.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10, strategy="quantile")
        plt.plot(prob_pred, prob_true, marker="o", linewidth=1.5, label=name)

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")
    plt.xlabel("Mean Predicted Probability", fontsize=12)
    plt.ylabel("Fraction of Positives (Empirical Fraud Rate)", fontsize=12)
    plt.title("Probability Calibration Curves", fontsize=14, fontweight="bold")
    plt.legend(loc="upper left", frameon=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info("Saved calibration curve to %s", output_path)


def plot_confusion_matrix_heatmap(
    cm_dict: Dict[str, int],
    threshold: float,
    model_name: str,
    output_path: Path = FIGURES_DIR / "confusion_matrix.png",
) -> None:
    """
    Renders styled confusion matrix heatmap.
    """
    cm = np.array([
        [cm_dict["true_negatives"], cm_dict["false_positives"]],
        [cm_dict["false_negatives"], cm_dict["true_positives"]],
    ])

    plt.figure(figsize=(7, 5.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Predicted Legit (0)", "Predicted Fraud (1)"],
        yticklabels=["Actual Legit (0)", "Actual Fraud (1)"],
        annot_kws={"size": 14, "fontweight": "bold"},
    )
    plt.title(f"Confusion Matrix: {model_name} (Threshold = {threshold:.3f})", fontsize=13, fontweight="bold")
    plt.ylabel("Actual Label", fontsize=11)
    plt.xlabel("Predicted Label", fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info("Saved confusion matrix heatmap to %s", output_path)


def run_full_evaluation() -> Dict[str, Any]:
    """
    Executes training suite, determines optimal thresholds on validation set,
    evaluates on out-of-time test set, persists best model artifact, and logs metrics.json.
    """
    suite = run_training_suite()
    data = suite["data"]
    models = suite["models"]

    X_train, y_train = data["train"]
    X_val, y_val = data["val"]
    X_test, y_test = data["test"]

    metrics_record: Dict[str, Any] = {
        "dataset_summary": {
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "test_samples": len(X_test),
            "train_fraud_count": int(y_train.sum()),
            "val_fraud_count": int(y_val.sum()),
            "test_fraud_count": int(y_test.sum()),
            "train_fraud_rate_pct": float((y_train.sum() / len(y_train)) * 100),
            "test_fraud_rate_pct": float((y_test.sum() / len(y_test)) * 100),
        },
        "models": {},
        "business_cost_assumptions": {
            "cost_false_negative_usd": COST_FALSE_NEGATIVE,
            "cost_false_positive_usd": COST_FALSE_POSITIVE,
            "target_recall": TARGET_RECALL,
        },
    }

    # Generate comparative curves on test set
    plot_precision_recall_curves(models, X_test, y_test)
    plot_roc_curves(models, X_test, y_test)
    plot_calibration_curves(models, X_test, y_test)

    best_model_name = None
    best_pr_auc = -1.0
    best_pipeline = None
    best_threshold = 0.5

    for name, model in models.items():
        logger.info("Evaluating %s...", name)

        # 1. Optimize threshold on Validation Set (Zero test leakage!)
        val_prob = model.predict_proba(X_val)[:, 1]
        threshold_info = find_optimal_thresholds(y_val, val_prob)
        chosen_thresh = threshold_info["cost_optimal_threshold"]

        # 2. Score on Test Set using chosen threshold and default 0.5
        test_prob = model.predict_proba(X_test)[:, 1]
        test_metrics_chosen = evaluate_probabilities(y_test, test_prob, threshold=chosen_thresh)
        test_metrics_default = evaluate_probabilities(y_test, test_prob, threshold=0.5)

        metrics_record["models"][name] = {
            "validation_threshold_optimization": threshold_info,
            "test_metrics_at_cost_optimal_threshold": test_metrics_chosen,
            "test_metrics_at_default_threshold": test_metrics_default,
        }

        if test_metrics_chosen["pr_auc"] > best_pr_auc:
            best_pr_auc = test_metrics_chosen["pr_auc"]
            best_model_name = name
            best_pipeline = model
            best_threshold = chosen_thresh

    logger.info("Best performing model: %s with Test PR-AUC = %.4f", best_model_name, best_pr_auc)

    # Save confusion matrix for best model
    best_cm = metrics_record["models"][best_model_name]["test_metrics_at_cost_optimal_threshold"]["confusion_matrix"]
    plot_confusion_matrix_heatmap(best_cm, best_threshold, best_model_name)

    # Persist Best Pipeline & Metadata
    metadata = {
        "best_model_name": best_model_name,
        "chosen_decision_threshold": best_threshold,
        "test_pr_auc": best_pr_auc,
        "features": FEATURE_COLS,
        "cost_assumptions": {
            "cost_false_negative": COST_FALSE_NEGATIVE,
            "cost_false_positive": COST_FALSE_POSITIVE,
        },
    }
    save_model_artifact(best_pipeline, metadata)

    # Write metrics.json programmatically
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_record, f, indent=2)
    logger.info("Metrics written programmatically to %s", METRICS_PATH)

    return metrics_record


if __name__ == "__main__":
    metrics = run_full_evaluation()
    print(f"\nEvaluation complete. Programmatic results written to {METRICS_PATH}")
