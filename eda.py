import os

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# ---------------------------------------------------
# Configuration
# ---------------------------------------------------

DATA_PATH = "train-test.csv"
OUTPUT_DIR = "EDA"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------
# Load Data
# ---------------------------------------------------

df = pd.read_csv(DATA_PATH)

print("="*60)
print("DATASET OVERVIEW")
print("="*60)

print(df.head())
print()

print(df.info())

print("\nShape:", df.shape)

# ---------------------------------------------------
# Missing Values
# ---------------------------------------------------

print("\n" + "="*60)
print("MISSING VALUES")
print("="*60)

missing = df.isnull().sum()
print(missing)

missing.to_csv(os.path.join(OUTPUT_DIR, "missing_values.csv"))

# ---------------------------------------------------
# Duplicate Rows
# ---------------------------------------------------

duplicates = df.duplicated().sum()

print("\nDuplicate Rows:", duplicates)

# ---------------------------------------------------
# Statistics
# ---------------------------------------------------

print("\n" + "="*60)
print("DESCRIPTIVE STATISTICS")
print("="*60)

stats = df.describe(include="all")

print(stats)

stats.to_csv(os.path.join(OUTPUT_DIR, "statistics.csv"))

# ---------------------------------------------------
# Numerical Features
# ---------------------------------------------------

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

# remove target if desired
target = "posted_rate"

# ---------------------------------------------------
# Correlation Matrix
# ---------------------------------------------------

corr = df[numeric_cols].corr()

plt.figure(figsize=(12,10))
plt.imshow(corr, cmap="coolwarm", interpolation="nearest")
plt.colorbar()
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "correlation_heatmap.png"))
plt.close()

# ---------------------------------------------------
# Histograms
# ---------------------------------------------------

for col in numeric_cols:

    plt.figure(figsize=(6,4))

    plt.hist(df[col], bins=40)

    plt.title(col)

    plt.tight_layout()

    plt.savefig(os.path.join(OUTPUT_DIR, f"hist_{col}.png"))

    plt.close()

# ---------------------------------------------------
# Boxplots (Outlier Detection)
# ---------------------------------------------------

for col in numeric_cols:

    plt.figure(figsize=(6,2))

    plt.boxplot(df[col], vert=False)

    plt.title(col)

    plt.tight_layout()

    plt.savefig(os.path.join(OUTPUT_DIR, f"boxplot_{col}.png"))

    plt.close()

# ---------------------------------------------------
# Scatter vs Target
# ---------------------------------------------------

if target in df.columns:

    features = [c for c in numeric_cols if c != target]

    for col in features:

        plt.figure(figsize=(5,4))

        plt.scatter(df[col], df[target], alpha=0.3)

        plt.xlabel(col)

        plt.ylabel(target)

        plt.tight_layout()

        plt.savefig(os.path.join(OUTPUT_DIR, f"scatter_{col}.png"))

        plt.close()

# ---------------------------------------------------
# Category Counts
# ---------------------------------------------------

categorical_cols = df.select_dtypes(include="object").columns

for col in categorical_cols:

    plt.figure(figsize=(8,4))

    df[col].value_counts().plot(kind="bar")

    plt.title(col)

    plt.tight_layout()

    plt.savefig(os.path.join(OUTPUT_DIR, f"category_{col}.png"))

    plt.close()

# ---------------------------------------------------
# IQR Outlier Report
# ---------------------------------------------------

print("\n" + "="*60)
print("OUTLIER REPORT")
print("="*60)

report = []

for col in numeric_cols:

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = ((df[col] < lower) | (df[col] > upper)).sum()

    report.append(
        {
            "Feature": col,
            "Lower Bound": lower,
            "Upper Bound": upper,
            "Outliers": outliers,
            "Percent": outliers / len(df) * 100
        }
    )

report = pd.DataFrame(report)

print(report)

report.to_csv(
    os.path.join(OUTPUT_DIR, "outlier_report.csv"),
    index=False
)

# ---------------------------------------------------
# Correlation with Target
# ---------------------------------------------------

if target in df.columns:

    print("\n" + "="*60)
    print("CORRELATION WITH TARGET")
    print("="*60)

    corr_target = (
        df[numeric_cols]
        .corr()[target]
        .sort_values(ascending=False)
    )

    print(corr_target)

    corr_target.to_csv(
        os.path.join(OUTPUT_DIR, "target_correlation.csv")
    )

print("\nEDA completed successfully.")
print(f"Results saved to: {OUTPUT_DIR}")