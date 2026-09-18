# STEP 7: INSIGHTS & REPORTING
# Clinical Trial Disease Category Classification Using NLP and Machine Learning

import os
import pandas as pd


# ---------------------------------------------------------
# 1. PATHS
# ---------------------------------------------------------

CLEANED_DATA_PATH = "clinical_trials_cleaned.csv"
CATEGORY_PATH = "disease_category_distribution.csv"
TERMS_PATH = "top_terms_overall.csv"
AVG_LENGTH_PATH = "average_summary_length_by_category.csv"
MODEL_PATH = "model_comparison.csv"
METADATA_PATH = "model_metadata.pkl"

OUTPUT_DIR = "insights"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------
# 2. LOAD DATA
# ---------------------------------------------------------

print("=" * 70)
print("STEP 7: INSIGHTS & REPORTING")
print("=" * 70)

df = pd.read_csv(CLEANED_DATA_PATH)

print("\nClinical trial dataset loaded.")
print("Shape:", df.shape)


# ---------------------------------------------------------
# 3. DISEASE CATEGORY DISTRIBUTION
# ---------------------------------------------------------

print("\n" + "-" * 70)
print("1. DISEASE CATEGORY DISTRIBUTION")
print("-" * 70)

category_counts = (
    df["disease_category"]
    .value_counts()
    .reset_index()
)

category_counts.columns = ["disease_category", "trial_count"]

total_trials = category_counts["trial_count"].sum()

category_counts["percentage"] = (
    category_counts["trial_count"] / total_trials * 100
).round(2)

category_counts.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "disease_category_insights.csv"
    ),
    index=False
)

print("\nDisease category distribution:")
print(category_counts.to_string(index=False))


# ---------------------------------------------------------
# 4. FREQUENT MEDICAL TERMS
# ---------------------------------------------------------

print("\n" + "-" * 70)
print("2. FREQUENTLY OCCURRING MEDICAL TERMS")
print("-" * 70)

if os.path.exists(TERMS_PATH):

    terms_df = pd.read_csv(TERMS_PATH)

    terms_df = terms_df.head(20)

else:

    # Fallback: calculate terms directly
    from collections import Counter

    all_tokens = (
        " ".join(
            df["cleaned_summary"]
            .fillna("")
            .astype(str)
        )
        .split()
    )

    top_terms = Counter(all_tokens).most_common(20)

    terms_df = pd.DataFrame(
        top_terms,
        columns=["term", "frequency"]
    )

terms_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "frequent_medical_terms.csv"
    ),
    index=False
)

print("\nTop frequently occurring terms:")
print(terms_df.to_string(index=False))


# ---------------------------------------------------------
# 5. CLINICAL TRIAL TEXT PATTERNS
# ---------------------------------------------------------

print("\n" + "-" * 70)
print("3. CLINICAL TRIAL TEXT PATTERNS AND TRENDS")
print("-" * 70)

df["brief_summary"] = (
    df["brief_summary"]
    .fillna("")
    .astype(str)
)

df["cleaned_summary"] = (
    df["cleaned_summary"]
    .fillna("")
    .astype(str)
)

df["summary_length"] = df["brief_summary"].str.len()

df["word_count"] = (
    df["cleaned_summary"]
    .str.split()
    .str.len()
)

overall_avg_length = df["summary_length"].mean()
overall_median_length = df["summary_length"].median()
overall_avg_words = df["word_count"].mean()
overall_median_words = df["word_count"].median()


text_pattern_data = pd.DataFrame({
    "metric": [
        "Total clinical trials",
        "Average summary length (characters)",
        "Median summary length (characters)",
        "Average cleaned summary word count",
        "Median cleaned summary word count",
        "Number of disease categories"
    ],
    "value": [
        len(df),
        round(overall_avg_length, 2),
        round(overall_median_length, 2),
        round(overall_avg_words, 2),
        round(overall_median_words, 2),
        df["disease_category"].nunique()
    ]
})

text_pattern_data.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "text_pattern_insights.csv"
    ),
    index=False
)

print("\nOverall text patterns:")
print(text_pattern_data.to_string(index=False))


# ---------------------------------------------------------
# 6. CATEGORY-LEVEL TEXT PATTERNS
# ---------------------------------------------------------

category_text_patterns = (
    df.groupby("disease_category")
    .agg(
        trial_count=("nct_id", "count"),
        average_summary_length=("summary_length", "mean"),
        median_summary_length=("summary_length", "median"),
        average_word_count=("word_count", "mean")
    )
    .reset_index()
)

category_text_patterns["average_summary_length"] = (
    category_text_patterns["average_summary_length"]
    .round(2)
)

category_text_patterns["median_summary_length"] = (
    category_text_patterns["median_summary_length"]
    .round(2)
)

category_text_patterns["average_word_count"] = (
    category_text_patterns["average_word_count"]
    .round(2)
)

category_text_patterns.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "category_text_patterns.csv"
    ),
    index=False
)

print("\nCategory-level text patterns:")
print(category_text_patterns.to_string(index=False))


# ---------------------------------------------------------
# 7. MODEL PERFORMANCE
# ---------------------------------------------------------

print("\n" + "-" * 70)
print("4. MODEL PERFORMANCE AND PREDICTION ACCURACY")
print("-" * 70)

model_df = pd.read_csv(MODEL_PATH)

model_df = model_df.sort_values(
    "f1_weighted",
    ascending=False
).reset_index(drop=True)

model_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "model_performance_insights.csv"
    ),
    index=False
)

print("\nModel performance:")
print(model_df.to_string(index=False))


# ---------------------------------------------------------
# 8. SELECTED MODEL
# ---------------------------------------------------------

selected_model = model_df.iloc[0]["model"]

selected_row = model_df.iloc[0]

accuracy = selected_row["accuracy"]
precision = selected_row["precision_weighted"]
recall = selected_row["recall_weighted"]
f1 = selected_row["f1_weighted"]

print("\nSelected model:", selected_model)
print("Accuracy:", round(accuracy, 4))
print("Weighted Precision:", round(precision, 4))
print("Weighted Recall:", round(recall, 4))
print("Weighted F1-score:", round(f1, 4))


# ---------------------------------------------------------
# 9. GENERATE FINAL TEXT REPORT
# ---------------------------------------------------------

report_path = os.path.join(
    OUTPUT_DIR,
    "final_insights_report.txt"
)

with open(report_path, "w", encoding="utf-8") as report:

    report.write(
        "CLINICAL TRIAL DISEASE CATEGORY CLASSIFICATION\n"
    )
    report.write(
        "INSIGHTS AND REPORTING\n"
    )
    report.write("=" * 70 + "\n\n")

    # Disease distribution
    report.write(
        "1. DISEASE CATEGORY DISTRIBUTION\n"
    )
    report.write("-" * 50 + "\n")

    report.write(
        f"Total clinical trials analyzed: {total_trials}\n"
    )

    report.write(
        f"Number of disease categories: "
        f"{df['disease_category'].nunique()}\n\n"
    )

    for _, row in category_counts.iterrows():

        report.write(
            f"- {row['disease_category']}: "
            f"{int(row['trial_count'])} trials "
            f"({row['percentage']:.2f}%)\n"
        )

    report.write("\n")

    # Frequent terms
    report.write(
        "2. FREQUENTLY OCCURRING MEDICAL TERMS\n"
    )
    report.write("-" * 50 + "\n")

    for _, row in terms_df.iterrows():

        report.write(
            f"- {row['term']}: "
            f"{row['frequency']}\n"
        )

    report.write("\n")

    # Text patterns
    report.write(
        "3. CLINICAL TRIAL TEXT PATTERNS AND TRENDS\n"
    )
    report.write("-" * 50 + "\n")

    report.write(
        f"Average summary length: "
        f"{overall_avg_length:.2f} characters\n"
    )

    report.write(
        f"Median summary length: "
        f"{overall_median_length:.2f} characters\n"
    )

    report.write(
        f"Average cleaned summary word count: "
        f"{overall_avg_words:.2f}\n"
    )

    report.write(
        f"Median cleaned summary word count: "
        f"{overall_median_words:.2f}\n"
    )

    report.write("\n")

    # Model performance
    report.write(
        "4. MODEL PERFORMANCE AND PREDICTION ACCURACY\n"
    )
    report.write("-" * 50 + "\n")

    for _, row in model_df.iterrows():

        report.write(
            f"\n{row['model']}\n"
        )

        report.write(
            f"  Accuracy: "
            f"{row['accuracy']:.4f}\n"
        )

        report.write(
            f"  Weighted Precision: "
            f"{row['precision_weighted']:.4f}\n"
        )

        report.write(
            f"  Weighted Recall: "
            f"{row['recall_weighted']:.4f}\n"
        )

        report.write(
            f"  Weighted F1-score: "
            f"{row['f1_weighted']:.4f}\n"
        )

    report.write("\n")

    report.write(
        f"Selected model based on highest weighted F1-score: "
        f"{selected_model}\n"
    )

    report.write("\n")

    # Decision-support statement
    report.write(
        "5. HEALTHCARE ANALYTICS AND RESEARCH SUPPORT\n"
    )
    report.write("-" * 50 + "\n")

    report.write(
        "The analysis provides a structured view of disease "
        "category distribution, frequently occurring medical "
        "terms, clinical-trial text characteristics, and "
        "machine-learning model performance.\n\n"
    )

    report.write(
        "The NLP classification pipeline can support clinical "
        "trial data organization, disease-category discovery, "
        "information retrieval, and research-oriented analytics. "
        "Predictions should be treated as analytical assistance "
        "and should not replace expert medical or research judgment.\n"
    )


# ---------------------------------------------------------
# 10. FINAL SUMMARY
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 7 COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nGenerated insight files:")

print(
    "1.",
    os.path.join(
        OUTPUT_DIR,
        "disease_category_insights.csv"
    )
)

print(
    "2.",
    os.path.join(
        OUTPUT_DIR,
        "frequent_medical_terms.csv"
    )
)

print(
    "3.",
    os.path.join(
        OUTPUT_DIR,
        "text_pattern_insights.csv"
    )
)

print(
    "4.",
    os.path.join(
        OUTPUT_DIR,
        "category_text_patterns.csv"
    )
)

print(
    "5.",
    os.path.join(
        OUTPUT_DIR,
        "model_performance_insights.csv"
    )
)

print(
    "6.",
    report_path
)

print("\nStep 7 reporting completed.")