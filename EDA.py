# ================================================================
# STEP 3: EXPLORATORY DATA ANALYSIS (EDA)
# Project: Clinical Trial Disease Category Classification
# ================================================================

import os
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

INPUT_PATH = "clinical_trials_cleaned.csv"
FIG_DIR = "figures"

os.makedirs(FIG_DIR, exist_ok=True)

# ---------------------------------------------------------------------
# Load cleaned dataset
# ---------------------------------------------------------------------

df = pd.read_csv(INPUT_PATH)

print("=" * 70)
print("STEP 3: EXPLORATORY DATA ANALYSIS")
print("=" * 70)

print("\nDataset shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

# Make sure text columns do not contain NaN
df["brief_summary"] = df["brief_summary"].fillna("").astype(str)
df["cleaned_summary"] = df["cleaned_summary"].fillna("").astype(str)
df["disease_category"] = df["disease_category"].fillna("").astype(str)

# ---------------------------------------------------------------------
# 1. Distribution of disease categories
# ---------------------------------------------------------------------

print("\n" + "=" * 70)
print("1. DISEASE CATEGORY DISTRIBUTION")
print("=" * 70)

category_counts = df["disease_category"].value_counts()

print(category_counts)

print("\nNumber of disease categories:")
print(df["disease_category"].nunique())

# Save category distribution
category_counts.to_csv(
    "disease_category_distribution.csv",
    header=["trial_count"]
)

plt.figure(figsize=(10, 6))

order = category_counts.index

sns.countplot(
    y="disease_category",
    data=df,
    order=order,
    hue="disease_category",
    palette="viridis",
    legend=False
)

plt.title("Clinical Trial Count by Disease Category")
plt.xlabel("Number of Trials")
plt.ylabel("Disease Category")
plt.tight_layout()

plt.savefig(
    f"{FIG_DIR}/category_distribution.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close()

# ---------------------------------------------------------------------
# 2. Statistical summary of text length
# ---------------------------------------------------------------------

print("\n" + "=" * 70)
print("2. SUMMARY LENGTH ANALYSIS")
print("=" * 70)

df["summary_length"] = df["brief_summary"].str.len()

print("\nSummary length statistics:")
print(df["summary_length"].describe())

print("\nAverage summary length:")
print(df["summary_length"].mean())

print("\nMedian summary length:")
print(df["summary_length"].median())

plt.figure(figsize=(9, 5))

sns.histplot(
    df["summary_length"],
    bins=50,
    kde=True
)

plt.title("Distribution of Brief Summary Length")
plt.xlabel("Character Count")
plt.ylabel("Number of Trials")
plt.tight_layout()

plt.savefig(
    f"{FIG_DIR}/summary_length_distribution.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close()

# ---------------------------------------------------------------------
# 3. Most common keywords overall
# ---------------------------------------------------------------------

print("\n" + "=" * 70)
print("3. TOP 20 MEDICAL TERMS OVERALL")
print("=" * 70)

all_tokens = (
    " ".join(df["cleaned_summary"].dropna())
    .split()
)

top_overall = Counter(all_tokens).most_common(20)

print("\nTop 20 overall terms:")
print(top_overall)

# Save top terms
top_overall_df = pd.DataFrame(
    top_overall,
    columns=["term", "frequency"]
)

top_overall_df.to_csv(
    "top_terms_overall.csv",
    index=False
)

if top_overall:

    words, counts = zip(*top_overall)

    plt.figure(figsize=(9, 6))

    sns.barplot(
        x=list(counts),
        y=list(words),
        hue=list(words),
        palette="mako",
        legend=False
    )

    plt.title("Top 20 Most Frequent Terms (All Categories)")
    plt.xlabel("Frequency")
    plt.ylabel("Medical Term")
    plt.tight_layout()

    plt.savefig(
        f"{FIG_DIR}/top_terms_overall.png",
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

# ---------------------------------------------------------------------
# 4. Most common keywords per disease category
# ---------------------------------------------------------------------

print("\n" + "=" * 70)
print("4. TOP TERMS BY DISEASE CATEGORY")
print("=" * 70)

categories = df["disease_category"].unique()

ncols = 2
nrows = (len(categories) + ncols - 1) // ncols

fig, axes = plt.subplots(
    nrows,
    ncols,
    figsize=(12, 4 * nrows)
)

# Convert axes into a flat array
if hasattr(axes, "flatten"):
    axes = axes.flatten()
else:
    axes = [axes]

for i, cat in enumerate(categories):

    category_text = df.loc[
        df["disease_category"] == cat,
        "cleaned_summary"
    ]

    tokens = " ".join(
        category_text.dropna()
    ).split()

    top = Counter(tokens).most_common(10)

    if top:

        words, counts = zip(*top)

        sns.barplot(
            x=list(counts),
            y=list(words),
            ax=axes[i],
            hue=list(words),
            palette="crest",
            legend=False
        )

    axes[i].set_title(str(cat))
    axes[i].set_xlabel("Frequency")
    axes[i].set_ylabel("Term")

# Remove unused subplot areas
for j in range(len(categories), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()

plt.savefig(
    f"{FIG_DIR}/top_terms_by_category.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close()

# ---------------------------------------------------------------------
# 5. Relationship: summary length vs disease category
# ---------------------------------------------------------------------

print("\n" + "=" * 70)
print("5. SUMMARY LENGTH BY DISEASE CATEGORY")
print("=" * 70)

print(
    df.groupby("disease_category")["summary_length"]
    .agg(["count", "mean", "median", "min", "max"])
    .sort_values("mean", ascending=False)
)

plt.figure(figsize=(12, 6))

sns.boxplot(
    data=df,
    x="disease_category",
    y="summary_length",
    hue="disease_category",
    palette="Set2",
    legend=False
)

plt.title("Summary Length by Disease Category")
plt.xlabel("Disease Category")
plt.ylabel("Summary Length (Characters)")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()

plt.savefig(
    f"{FIG_DIR}/summary_length_by_category.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close()

# ---------------------------------------------------------------------
# 6. Average summary length by disease category
# ---------------------------------------------------------------------

avg_length = (
    df.groupby("disease_category")["summary_length"]
    .mean()
    .sort_values(ascending=False)
)

avg_length.to_csv(
    "average_summary_length_by_category.csv",
    header=["average_summary_length"]
)

plt.figure(figsize=(12, 6))

sns.barplot(
    x=avg_length.values,
    y=avg_length.index,
    hue=avg_length.index,
    palette="rocket",
    legend=False
)

plt.title("Average Clinical Trial Summary Length by Disease Category")
plt.xlabel("Average Character Count")
plt.ylabel("Disease Category")
plt.tight_layout()

plt.savefig(
    f"{FIG_DIR}/average_summary_length_by_category.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close()

# ---------------------------------------------------------------------
# 7. Save EDA summary
# ---------------------------------------------------------------------

eda_summary = pd.DataFrame({
    "metric": [
        "Total clinical trials",
        "Unique disease categories",
        "Average summary length",
        "Median summary length",
        "Minimum summary length",
        "Maximum summary length"
    ],
    "value": [
        len(df),
        df["disease_category"].nunique(),
        df["summary_length"].mean(),
        df["summary_length"].median(),
        df["summary_length"].min(),
        df["summary_length"].max()
    ]
})

eda_summary.to_csv(
    "eda_summary.csv",
    index=False
)

# ---------------------------------------------------------------------
# 8. Final output
# ---------------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 3 COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nEDA output files:")

print("CSV files:")
print("  - disease_category_distribution.csv")
print("  - top_terms_overall.csv")
print("  - average_summary_length_by_category.csv")
print("  - eda_summary.csv")

print("\nFigure files:")
print(f"  - {FIG_DIR}/category_distribution.png")
print(f"  - {FIG_DIR}/summary_length_distribution.png")
print(f"  - {FIG_DIR}/top_terms_overall.png")
print(f"  - {FIG_DIR}/top_terms_by_category.png")
print(f"  - {FIG_DIR}/summary_length_by_category.png")
print(f"  - {FIG_DIR}/average_summary_length_by_category.png")

print("\nEDA analysis completed.")