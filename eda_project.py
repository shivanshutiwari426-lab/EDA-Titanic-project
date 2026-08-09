"""
Exploratory Data Analysis (EDA) Project
========================================
Goal: Analyze the Titanic passenger dataset to uncover patterns and trends,
and identify which factors most influenced passenger survival.

Dataset: Titanic passenger manifest (891 passengers, 15 fields — mix of
numeric and categorical, with real missing data). Loaded directly from the
public seaborn-data repository, no local file needed.

This script produces:
  - Console statistical summaries (describe, missing values, value counts)
  - A correlation heatmap
  - Univariate distribution plots (age, fare)
  - Bivariate plots showing survival broken down by sex, class, age, fare
  - A multivariate plot combining several factors at once
  - eda_summary_stats.csv — key numbers referenced in the written report
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 20)

OUT = "/home/claude"

# ---------------------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------------------
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
df = pd.read_csv(url)

print("=" * 70)
print("DATASET OVERVIEW")
print("=" * 70)
print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns\n")
print("First 5 rows:")
print(df.head(), "\n")
print("Data types:")
print(df.dtypes, "\n")

# ---------------------------------------------------------------------------
# 2. STATISTICAL SUMMARY
# ---------------------------------------------------------------------------
print("=" * 70)
print("STATISTICAL SUMMARY (numeric columns)")
print("=" * 70)
print(df.describe(), "\n")

print("=" * 70)
print("MISSING VALUES")
print("=" * 70)
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(1)
missing_df = pd.DataFrame({"missing_count": missing, "missing_pct": missing_pct})
missing_df = missing_df[missing_df["missing_count"] > 0].sort_values("missing_count", ascending=False)
print(missing_df, "\n")

# ---------------------------------------------------------------------------
# 3. CLEANING (light touch, for visualization purposes)
# ---------------------------------------------------------------------------
df_clean = df.copy()
df_clean["age"] = df_clean["age"].fillna(df_clean["age"].median())
df_clean["embark_town"] = df_clean["embark_town"].fillna("Unknown")
df_clean["deck"] = df_clean["deck"].astype(str).fillna("Unknown")

# ---------------------------------------------------------------------------
# 4. KEY GROUP-LEVEL STATS
# ---------------------------------------------------------------------------
survival_rate_overall = df_clean["survived"].mean()
survival_by_sex = df_clean.groupby("sex")["survived"].mean()
survival_by_class = df_clean.groupby("pclass")["survived"].mean()
survival_by_embark = df_clean.groupby("embark_town")["survived"].mean()

print("=" * 70)
print("SURVIVAL RATE BREAKDOWNS")
print("=" * 70)
print(f"Overall survival rate: {survival_rate_overall:.1%}\n")
print("By sex:")
print(survival_by_sex, "\n")
print("By passenger class:")
print(survival_by_class, "\n")
print("By embarkation town:")
print(survival_by_embark, "\n")

# ---------------------------------------------------------------------------
# 5. CORRELATION ANALYSIS
# ---------------------------------------------------------------------------
corr_df = df_clean.copy()
corr_df["sex_num"] = corr_df["sex"].map({"male": 0, "female": 1})
corr_df["alone_num"] = corr_df["alone"].astype(int)
numeric_cols = ["survived", "pclass", "sex_num", "age", "sibsp", "parch", "fare", "alone_num"]
corr_matrix = corr_df[numeric_cols].corr()

print("=" * 70)
print("CORRELATION WITH SURVIVAL (sorted)")
print("=" * 70)
survival_corr = corr_matrix["survived"].drop("survived").sort_values(key=abs, ascending=False)
print(survival_corr, "\n")

# Save key stats to CSV for reference in the report
summary_export = pd.DataFrame({
    "correlation_with_survival": survival_corr
}).round(3)
summary_export.to_csv(f"{OUT}/eda_summary_stats.csv")
print(f"Saved: eda_summary_stats.csv")

# ---------------------------------------------------------------------------
# 6. VISUALIZATION 1 — Correlation Heatmap
# ---------------------------------------------------------------------------
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0,
            square=True, linewidths=0.5)
plt.title("Correlation Heatmap — Numeric Features")
plt.tight_layout()
plt.savefig(f"{OUT}/eda_correlation_heatmap.png", dpi=150)
plt.close()
print("Saved: eda_correlation_heatmap.png")

# ---------------------------------------------------------------------------
# 7. VISUALIZATION 2 — Age & Fare Distributions
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
sns.histplot(df_clean["age"], bins=30, kde=True, ax=axes[0], color="steelblue")
axes[0].set_title("Age Distribution")
axes[0].set_xlabel("Age")

sns.histplot(df_clean["fare"], bins=40, kde=True, ax=axes[1], color="seagreen")
axes[1].set_title("Fare Distribution")
axes[1].set_xlabel("Fare ($)")
plt.tight_layout()
plt.savefig(f"{OUT}/eda_distributions.png", dpi=150)
plt.close()
print("Saved: eda_distributions.png")

# ---------------------------------------------------------------------------
# 8. VISUALIZATION 3 — Survival by Sex, Class, Embarkation
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

sns.barplot(data=df_clean, x="sex", y="survived", ax=axes[0], palette="Set2")
axes[0].set_title("Survival Rate by Sex")
axes[0].set_ylabel("Survival Rate")
axes[0].set_ylim(0, 1)

sns.barplot(data=df_clean, x="pclass", y="survived", ax=axes[1], palette="Set2")
axes[1].set_title("Survival Rate by Passenger Class")
axes[1].set_ylabel("Survival Rate")
axes[1].set_ylim(0, 1)

sns.barplot(data=df_clean, x="embark_town", y="survived", ax=axes[2], palette="Set2")
axes[2].set_title("Survival Rate by Embarkation Town")
axes[2].set_ylabel("Survival Rate")
axes[2].set_ylim(0, 1)
axes[2].tick_params(axis="x", rotation=20)

plt.tight_layout()
plt.savefig(f"{OUT}/eda_survival_breakdown.png", dpi=150)
plt.close()
print("Saved: eda_survival_breakdown.png")

# ---------------------------------------------------------------------------
# 9. VISUALIZATION 4 — Age vs Fare vs Survival vs Class (multivariate)
# ---------------------------------------------------------------------------
plt.figure(figsize=(9, 6))
sns.scatterplot(
    data=df_clean, x="age", y="fare", hue="survived", style="pclass",
    palette={0: "indianred", 1: "seagreen"}, alpha=0.7, s=70
)
plt.title("Age vs Fare, colored by Survival, styled by Class")
plt.xlabel("Age")
plt.ylabel("Fare ($)")
plt.legend(title="Survived / Class", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig(f"{OUT}/eda_multivariate_scatter.png", dpi=150)
plt.close()
print("Saved: eda_multivariate_scatter.png")

# ---------------------------------------------------------------------------
# 10. VISUALIZATION 5 — Fare distribution by class (boxplot)
# ---------------------------------------------------------------------------
plt.figure(figsize=(8, 5))
sns.boxplot(data=df_clean, x="pclass", y="fare", palette="Set3")
plt.title("Fare Distribution by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Fare ($)")
plt.tight_layout()
plt.savefig(f"{OUT}/eda_fare_by_class_boxplot.png", dpi=150)
plt.close()
print("Saved: eda_fare_by_class_boxplot.png")

print("\nEDA script complete. See eda_summary_stats.csv and PNG files for report inputs.")
