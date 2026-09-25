# ================================================================
# STEP 7: INSIGHTS & REPORTING
# Project: Clinical Trial Disease Category Classification
# Using NLP and Machine Learning
# ================================================================

import os
import joblib
import pandas as pd


# ----------------------------------------------------------------
# 1. PATHS
# ----------------------------------------------------------------

CLEANED_DATA_PATH = "clinical_trials_cleaned.csv"
TERMS_PATH = "top_terms_overall.csv"
MODEL_PATH = "model_comparison.csv"
METADATA_PATH = "model_metadata.pkl"

OUTPUT_DIR = "insights"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ----------------------------------------------------------------
# 2. LOAD DATA
# ----------------------------------------------------------------

print("=" * 70)
print("STEP 7: INSIGHTS & REPORTING")
print("=" * 70)

df = pd.read_csv(
    CLEANED_DATA_PATH
)

print("\nClinical trial dataset loaded.")

print(
    "Shape:",
    df.shape
)


# ----------------------------------------------------------------
# 3. CHECK REQUIRED COLUMNS
# ----------------------------------------------------------------

required_columns = [
    "nct_id",
    "brief_summary",
    "cleaned_summary",
    "disease_category"
]

for column in required_columns:

    if column not in df.columns:

        raise ValueError(
            f"Required column '{column}' "
            f"was not found in the dataset."
        )


# ----------------------------------------------------------------
# 4. CLEAN REQUIRED COLUMNS
# ----------------------------------------------------------------

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

df["disease_category"] = (
    df["disease_category"]
    .fillna("")
    .astype(str)
)

# Remove empty disease-category records
df = df[
    df["disease_category"].str.strip() != ""
].copy()

df = df.reset_index(
    drop=True
)


# ----------------------------------------------------------------
# 5. DISEASE CATEGORY DISTRIBUTION
# ----------------------------------------------------------------

print("\n" + "-" * 70)
print("1. DISEASE CATEGORY DISTRIBUTION")
print("-" * 70)

category_counts = (
    df["disease_category"]
    .value_counts()
    .reset_index()
)

category_counts.columns = [
    "disease_category",
    "trial_count"
]

total_trials = (
    category_counts["trial_count"]
    .sum()
)

category_counts["percentage"] = (
    category_counts["trial_count"]
    / total_trials
    * 100
).round(2)


# Save disease category insights
category_output_path = os.path.join(
    OUTPUT_DIR,
    "disease_category_insights.csv"
)

category_counts.to_csv(
    category_output_path,
    index=False
)

print(
    "\nDisease category distribution:"
)

print(
    category_counts.to_string(
        index=False
    )
)


# ----------------------------------------------------------------
# 6. FREQUENTLY OCCURRING MEDICAL TERMS
# ----------------------------------------------------------------

print("\n" + "-" * 70)
print("2. FREQUENTLY OCCURRING MEDICAL TERMS")
print("-" * 70)


if os.path.exists(TERMS_PATH):

    terms_df = pd.read_csv(
        TERMS_PATH
    )

    terms_df = terms_df.head(
        20
    )

else:

    # ------------------------------------------------------------
    # Fallback:
    # Calculate frequent terms directly from cleaned text
    # ------------------------------------------------------------

    from collections import Counter

    all_tokens = (
        " ".join(
            df["cleaned_summary"]
            .fillna("")
            .astype(str)
        )
        .split()
    )

    top_terms = Counter(
        all_tokens
    ).most_common(20)

    terms_df = pd.DataFrame(
        top_terms,
        columns=[
            "term",
            "frequency"
        ]
    )


# Save frequent terms
terms_output_path = os.path.join(
    OUTPUT_DIR,
    "frequent_medical_terms.csv"
)

terms_df.to_csv(
    terms_output_path,
    index=False
)


print(
    "\nTop frequently occurring medical terms:"
)

print(
    terms_df.to_string(
        index=False
    )
)


# ----------------------------------------------------------------
# 7. CLINICAL TRIAL TEXT PATTERNS
# ----------------------------------------------------------------

print("\n" + "-" * 70)
print("3. CLINICAL TRIAL TEXT PATTERNS AND TRENDS")
print("-" * 70)


# Original summary length
df["summary_length"] = (
    df["brief_summary"]
    .str.len()
)


# Cleaned summary word count
df["word_count"] = (
    df["cleaned_summary"]
    .str.split()
    .str.len()
)


# Overall statistics
overall_avg_length = (
    df["summary_length"]
    .mean()
)

overall_median_length = (
    df["summary_length"]
    .median()
)

overall_avg_words = (
    df["word_count"]
    .mean()
)

overall_median_words = (
    df["word_count"]
    .median()
)


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

        round(
            overall_avg_length,
            2
        ),

        round(
            overall_median_length,
            2
        ),

        round(
            overall_avg_words,
            2
        ),

        round(
            overall_median_words,
            2
        ),

        df[
            "disease_category"
        ].nunique()
    ]
})


# Save overall text patterns
text_pattern_output_path = os.path.join(
    OUTPUT_DIR,
    "text_pattern_insights.csv"
)

text_pattern_data.to_csv(
    text_pattern_output_path,
    index=False
)


print(
    "\nOverall text patterns:"
)

print(
    text_pattern_data.to_string(
        index=False
    )
)


# ----------------------------------------------------------------
# 8. CATEGORY-LEVEL TEXT PATTERNS
# ----------------------------------------------------------------

print("\n" + "-" * 70)
print("CATEGORY-LEVEL TEXT PATTERNS")
print("-" * 70)


category_text_patterns = (

    df.groupby(
        "disease_category"
    )

    .agg(

        trial_count=(
            "nct_id",
            "count"
        ),

        average_summary_length=(
            "summary_length",
            "mean"
        ),

        median_summary_length=(
            "summary_length",
            "median"
        ),

        average_word_count=(
            "word_count",
            "mean"
        )
    )

    .reset_index()
)


category_text_patterns[
    "average_summary_length"
] = (

    category_text_patterns[
        "average_summary_length"
    ]
    .round(2)
)


category_text_patterns[
    "median_summary_length"
] = (

    category_text_patterns[
        "median_summary_length"
    ]
    .round(2)
)


category_text_patterns[
    "average_word_count"
] = (

    category_text_patterns[
        "average_word_count"
    ]
    .round(2)
)


category_pattern_output_path = os.path.join(
    OUTPUT_DIR,
    "category_text_patterns.csv"
)

category_text_patterns.to_csv(
    category_pattern_output_path,
    index=False
)


print(
    "\nCategory-level text patterns:"
)

print(
    category_text_patterns.to_string(
        index=False
    )
)


# ----------------------------------------------------------------
# 9. LOAD MODEL PERFORMANCE
# ----------------------------------------------------------------

print("\n" + "-" * 70)
print("4. MODEL PERFORMANCE AND PREDICTION ACCURACY")
print("-" * 70)


if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        f"{MODEL_PATH} was not found. "
        "Please run Step 6 Model Evaluation first."
    )


model_df = pd.read_csv(
    MODEL_PATH
)


# ----------------------------------------------------------------
# 10. CHECK MODEL METRIC COLUMNS
# ----------------------------------------------------------------

required_metric_columns = [

    "model",
    "accuracy",

    "precision_weighted",
    "recall_weighted",
    "f1_weighted",

    "precision_macro",
    "recall_macro",
    "f1_macro"
]


for column in required_metric_columns:

    if column not in model_df.columns:

        raise ValueError(
            f"Required metric column "
            f"'{column}' was not found in "
            f"{MODEL_PATH}. "
            f"Please run the corrected Step 6 first."
        )


# ----------------------------------------------------------------
# 11. SORT MODEL PERFORMANCE USING MACRO F1
# ----------------------------------------------------------------
# Macro F1 gives equal importance to every disease category.
# This is useful because the dataset is imbalanced.
# ----------------------------------------------------------------

model_df = model_df.sort_values(
    by="f1_macro",
    ascending=False
).reset_index(
    drop=True
)


model_performance_output_path = os.path.join(
    OUTPUT_DIR,
    "model_performance_insights.csv"
)


model_df.to_csv(
    model_performance_output_path,
    index=False
)


print(
    "\nModel performance:"
)

print(
    model_df.to_string(
        index=False
    )
)


# ----------------------------------------------------------------
# 12. LOAD MODEL METADATA
# ----------------------------------------------------------------

model_metadata = None

if os.path.exists(
    METADATA_PATH
):

    model_metadata = joblib.load(
        METADATA_PATH
    )

    print(
        "\nModel metadata loaded successfully."
    )

else:

    print(
        "\nWarning: model_metadata.pkl "
        "was not found."
    )


# ----------------------------------------------------------------
# 13. SELECTED MODEL
# ----------------------------------------------------------------

selected_model = (
    model_df.iloc[0]["model"]
)

selected_row = (
    model_df.iloc[0]
)


accuracy = (
    selected_row["accuracy"]
)

precision_weighted = (
    selected_row[
        "precision_weighted"
    ]
)

recall_weighted = (
    selected_row[
        "recall_weighted"
    ]
)

f1_weighted = (
    selected_row[
        "f1_weighted"
    ]
)

precision_macro = (
    selected_row[
        "precision_macro"
    ]
)

recall_macro = (
    selected_row[
        "recall_macro"
    ]
)

f1_macro = (
    selected_row[
        "f1_macro"
    ]
)


print("\n" + "-" * 70)
print("SELECTED MODEL")
print("-" * 70)


print(
    "\nSelected Model:",
    selected_model
)

print(
    "Selection Metric: Macro F1 Score"
)

print(
    "Accuracy:",
    round(
        accuracy,
        4
    )
)

print(
    "Weighted Precision:",
    round(
        precision_weighted,
        4
    )
)

print(
    "Weighted Recall:",
    round(
        recall_weighted,
        4
    )
)

print(
    "Weighted F1-score:",
    round(
        f1_weighted,
        4
    )
)

print(
    "Macro Precision:",
    round(
        precision_macro,
        4
    )
)

print(
    "Macro Recall:",
    round(
        recall_macro,
        4
    )
)

print(
    "Macro F1-score:",
    round(
        f1_macro,
        4
    )
)


# ----------------------------------------------------------------
# 14. VERIFY SELECTED MODEL AGAINST METADATA
# ----------------------------------------------------------------

if model_metadata is not None:

    metadata_model = model_metadata.get(
        "selected_model"
    )

    if metadata_model:

        print(
            "\nSelected model from metadata:",
            metadata_model
        )

        if metadata_model != selected_model:

            print(
                "\nWARNING:"
            )

            print(
                "Selected model in model_comparison.csv "
                "does not match model_metadata.pkl."
            )

            print(
                "Please rerun Step 6 before continuing."
            )


# ----------------------------------------------------------------
# 15. GENERATE FINAL TEXT REPORT
# ----------------------------------------------------------------

report_path = os.path.join(
    OUTPUT_DIR,
    "final_insights_report.txt"
)


with open(
    report_path,
    "w",
    encoding="utf-8"
) as report:


    # ============================================================
    # REPORT HEADER
    # ============================================================

    report.write(
        "CLINICAL TRIAL DISEASE CATEGORY CLASSIFICATION\n"
    )

    report.write(
        "INSIGHTS AND REPORTING\n"
    )

    report.write(
        "=" * 70
        + "\n\n"
    )


    # ============================================================
    # 1. DISEASE CATEGORY DISTRIBUTION
    # ============================================================

    report.write(
        "1. DISEASE CATEGORY DISTRIBUTION\n"
    )

    report.write(
        "-" * 50
        + "\n"
    )

    report.write(
        f"Total clinical trials analyzed: "
        f"{total_trials}\n"
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


    report.write(
        "\n"
    )


    # ============================================================
    # 2. FREQUENT MEDICAL TERMS
    # ============================================================

    report.write(
        "2. FREQUENTLY OCCURRING MEDICAL TERMS\n"
    )

    report.write(
        "-" * 50
        + "\n"
    )


    for _, row in terms_df.iterrows():

        # Support either frequency or mean_tfidf column
        if "frequency" in terms_df.columns:

            value = row[
                "frequency"
            ]

            metric_name = (
                "Frequency"
            )

        elif "mean_tfidf" in terms_df.columns:

            value = row[
                "mean_tfidf"
            ]

            metric_name = (
                "Mean TF-IDF"
            )

        else:

            value = ""

            metric_name = (
                "Value"
            )


        report.write(

            f"- {row['term']}: "

            f"{metric_name} = "

            f"{value}\n"
        )


    report.write(
        "\n"
    )


    # ============================================================
    # 3. CLINICAL TRIAL TEXT PATTERNS
    # ============================================================

    report.write(
        "3. CLINICAL TRIAL TEXT PATTERNS AND TRENDS\n"
    )

    report.write(
        "-" * 50
        + "\n"
    )


    report.write(

        f"Average summary length: "

        f"{overall_avg_length:.2f} "

        f"characters\n"
    )


    report.write(

        f"Median summary length: "

        f"{overall_median_length:.2f} "

        f"characters\n"
    )


    report.write(

        f"Average cleaned summary word count: "

        f"{overall_avg_words:.2f}\n"
    )


    report.write(

        f"Median cleaned summary word count: "

        f"{overall_median_words:.2f}\n"
    )


    report.write(
        "\n"
    )


    # ============================================================
    # 4. CATEGORY-LEVEL TEXT PATTERNS
    # ============================================================

    report.write(
        "4. CATEGORY-LEVEL TEXT PATTERNS\n"
    )

    report.write(
        "-" * 50
        + "\n"
    )


    for _, row in category_text_patterns.iterrows():

        report.write(

            f"\n{row['disease_category']}\n"
        )

        report.write(

            f"  Trial Count: "
            f"{int(row['trial_count'])}\n"
        )

        report.write(

            f"  Average Summary Length: "
            f"{row['average_summary_length']:.2f}\n"
        )

        report.write(

            f"  Median Summary Length: "
            f"{row['median_summary_length']:.2f}\n"
        )

        report.write(

            f"  Average Word Count: "
            f"{row['average_word_count']:.2f}\n"
        )


    report.write(
        "\n"
    )


    # ============================================================
    # 5. MODEL PERFORMANCE
    # ============================================================

    report.write(
        "5. MODEL PERFORMANCE AND PREDICTION ACCURACY\n"
    )

    report.write(
        "-" * 50
        + "\n"
    )


    for _, row in model_df.iterrows():

        report.write(

            f"\n{row['model']}\n"
        )


        report.write(

            f"  Accuracy: "

            f"{row['accuracy']:.4f}\n"
        )


        # --------------------------------------------------------
        # Weighted Metrics
        # --------------------------------------------------------

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


        # --------------------------------------------------------
        # Macro Metrics
        # --------------------------------------------------------

        report.write(

            f"  Macro Precision: "

            f"{row['precision_macro']:.4f}\n"
        )


        report.write(

            f"  Macro Recall: "

            f"{row['recall_macro']:.4f}\n"
        )


        report.write(

            f"  Macro F1-score: "

            f"{row['f1_macro']:.4f}\n"
        )


    report.write(
        "\n"
    )


    # ============================================================
    # 6. SELECTED MODEL
    # ============================================================

    report.write(
        "6. SELECTED MODEL\n"
    )

    report.write(
        "-" * 50
        + "\n"
    )


    report.write(

        f"Selected model based on highest "

        f"Macro F1-score: "

        f"{selected_model}\n\n"
    )


    report.write(

        f"Accuracy: "
        f"{accuracy:.4f}\n"
    )


    report.write(

        f"Weighted Precision: "
        f"{precision_weighted:.4f}\n"
    )


    report.write(

        f"Weighted Recall: "
        f"{recall_weighted:.4f}\n"
    )


    report.write(

        f"Weighted F1-score: "
        f"{f1_weighted:.4f}\n"
    )


    report.write(

        f"Macro Precision: "
        f"{precision_macro:.4f}\n"
    )


    report.write(

        f"Macro Recall: "
        f"{recall_macro:.4f}\n"
    )


    report.write(

        f"Macro F1-score: "
        f"{f1_macro:.4f}\n"
    )


    report.write(
        "\n"
    )


    # ============================================================
    # 7. HEALTHCARE ANALYTICS AND RESEARCH SUPPORT
    # ============================================================

    report.write(
        "7. HEALTHCARE ANALYTICS AND RESEARCH SUPPORT\n"
    )

    report.write(
        "-" * 50
        + "\n"
    )


    report.write(

        "The analysis provides a structured view of disease "
        "category distribution, frequently occurring medical "
        "terms, clinical-trial text characteristics, "
        "disease-specific text patterns, and machine-learning "
        "model performance.\n\n"
    )


    report.write(

        "The NLP classification pipeline can support clinical "
        "trial data organization, disease-category discovery, "
        "information retrieval, and research-oriented analytics. "
        "Predictions should be treated as analytical assistance "
        "and should not replace expert medical or research "
        "judgment.\n"
    )


# ----------------------------------------------------------------
# 16. FINAL SUMMARY
# ----------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 7 COMPLETED SUCCESSFULLY")
print("=" * 70)


print(
    "\nGenerated insight files:"
)


print(
    "1.",
    category_output_path
)

print(
    "2.",
    terms_output_path
)

print(
    "3.",
    text_pattern_output_path
)

print(
    "4.",
    category_pattern_output_path
)

print(
    "5.",
    model_performance_output_path
)

print(
    "6.",
    report_path
)


print(
    "\nSelected Model:"
)

print(
    selected_model
)


print(
    "\nSelection Metric:"
)

print(
    "Macro F1 Score"
)


print(
    "\nSelected Model Performance:"
)

print(
    f"Accuracy          : "
    f"{accuracy:.4f}"
)

print(
    f"Weighted F1-score : "
    f"{f1_weighted:.4f}"
)

print(
    f"Macro F1-score    : "
    f"{f1_macro:.4f}"
)


print(
    "\nStep 7 Insights & Reporting "
    "completed successfully."
)
