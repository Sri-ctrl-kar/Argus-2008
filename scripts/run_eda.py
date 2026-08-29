"""
Executable script for generating EDA figures.
"""

import sys
from pathlib import Path

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

from src.config import FIGURES_DIR, TARGET_COL, TIME_COL, AMOUNT_COL, PCA_FEATURE_COLS
from src.data import load_raw

sns.set_theme(style="whitegrid", palette="muted")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

print("Loading raw transaction data for EDA...")
df = load_raw()
print(f"Total Transactions: {len(df):,}")
print(f"Total Features: {df.shape[1]}")

# 1. Missingness & Duplicates
null_counts = df.isnull().sum()
print(f"Total Missing Values across all columns: {null_counts.sum()}")
duplicate_count = df.duplicated().sum()
print(f"Duplicate Rows: {duplicate_count:,} ({duplicate_count / len(df) * 100:.2f}%)")

# 2. Class Balance
class_counts = df[TARGET_COL].value_counts()
n_legit = class_counts[0]
n_fraud = class_counts[1]
fraud_pct = (n_fraud / len(df)) * 100
imbalance_ratio = n_legit / n_fraud

print("=" * 50)
print("CLASS DISTRIBUTION BREAKDOWN:")
print(f"Legitimate Transactions (0): {n_legit:,} ({100 - fraud_pct:.3f}%)")
print(f"Fraudulent Transactions (1): {n_fraud:,} ({fraud_pct:.3f}%)")
print(f"Class Imbalance Ratio:        {imbalance_ratio:.1f} : 1 (1 fraud per {imbalance_ratio:.0f} legit)")
print("=" * 50)

fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(["Legitimate (0)", "Fraud (1)"], [n_legit, n_fraud], color=["#2b5c8f", "#d9534f"])
ax.set_yscale("log")
ax.set_ylabel("Count (Log Scale)", fontsize=12)
ax.set_title(f"Class Imbalance: {fraud_pct:.3f}% Fraud ({imbalance_ratio:.0f}:1 Ratio)", fontsize=13, fontweight="bold")
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval * 1.3, f"{int(yval):,}", ha="center", va="bottom", fontweight="bold")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "class_distribution.png", dpi=300)
plt.close()
print(f"Saved {FIGURES_DIR / 'class_distribution.png'}")

# 3. Temporal Distribution
df_time = df.copy()
df_time["Hour"] = (df_time[TIME_COL] / 3600).astype(int)

hourly_stats = df_time.groupby("Hour").agg(
    total_tx=("Class", "count"),
    fraud_tx=("Class", "sum"),
).reset_index()
hourly_stats["fraud_rate_pct"] = (hourly_stats["fraud_tx"] / hourly_stats["total_tx"]) * 100

fig, ax1 = plt.subplots(figsize=(12, 5))
color = "tab:blue"
ax1.set_xlabel("Time Elapsed (Hours)", fontsize=12)
ax1.set_ylabel("Total Transaction Volume", color=color, fontsize=12)
ax1.plot(hourly_stats["Hour"], hourly_stats["total_tx"], color=color, linewidth=2, label="Total Volume")
ax1.tick_params(axis="y", labelcolor=color)

ax2 = ax1.twinx()
color = "tab:red"
ax2.set_ylabel("Fraud Rate (%)", color=color, fontsize=12)
ax2.plot(hourly_stats["Hour"], hourly_stats["fraud_rate_pct"], color=color, linestyle="--", marker="o", linewidth=2, label="Fraud Rate (%)")
ax2.tick_params(axis="y", labelcolor=color)

plt.title("Transaction Volume vs. Fraud Rate Over 48 Hours", fontsize=14, fontweight="bold")
fig.tight_layout()
plt.savefig(FIGURES_DIR / "fraud_rate_over_time.png", dpi=300)
plt.close()
print(f"Saved {FIGURES_DIR / 'fraud_rate_over_time.png'}")

# 4. Top Divergent Features
mean_diffs = {}
for col in PCA_FEATURE_COLS:
    legit_mean = df[df[TARGET_COL] == 0][col].mean()
    fraud_mean = df[df[TARGET_COL] == 1][col].mean()
    mean_diffs[col] = abs(fraud_mean - legit_mean)

top_divergent_features = sorted(mean_diffs.items(), key=lambda x: x[1], reverse=True)[:6]
top_feature_names = [f[0] for f in top_divergent_features]
print("Top 6 Most Divergent PCA Features:", top_divergent_features)

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()

for i, feature in enumerate(top_feature_names):
    sns.boxplot(x=TARGET_COL, y=feature, data=df, ax=axes[i], palette=["#2b5c8f", "#d9534f"], showfliers=False)
    axes[i].set_title(f"Distribution of {feature}", fontsize=12, fontweight="bold")
    axes[i].set_xticklabels(["Legit (0)", "Fraud (1)"])

plt.suptitle("Feature Distribution Comparison: Legit vs Fraud (Top Predictors)", fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "top_features_distribution.png", dpi=300)
plt.close()
print(f"Saved {FIGURES_DIR / 'top_features_distribution.png'}")

# 5. Amount Distribution
fig, ax = plt.subplots(figsize=(9, 5))
sns.kdeplot(np.log1p(df[df[TARGET_COL] == 0][AMOUNT_COL]), label="Legitimate (0)", fill=True, color="#2b5c8f", alpha=0.4)
sns.kdeplot(np.log1p(df[df[TARGET_COL] == 1][AMOUNT_COL]), label="Fraud (1)", fill=True, color="#d9534f", alpha=0.4)
ax.set_xlabel("log1p(Amount)", fontsize=12)
ax.set_ylabel("Density", fontsize=12)
ax.set_title("Distribution of log1p(Transaction Amount)", fontsize=13, fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig(FIGURES_DIR / "amount_distribution.png", dpi=300)
plt.close()
print(f"Saved {FIGURES_DIR / 'amount_distribution.png'}")

print("\nEDA figure generation completed successfully.")
