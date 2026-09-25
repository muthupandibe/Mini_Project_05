# ================================================================
# STEP 6: MODEL EVALUATION
# Project: Clinical Trial Disease Category Classification
# ================================================================

import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ----------------------------------------------------------------
# 1. CONFIGURATION
# ----------------------------------------------------------------

LOGISTIC_PATH = "logistic_regression.pkl"
NB_PATH = "multinomial_naive_bayes.pkl"
SVM_PATH = "linear_svm.pkl"

ENCODER_PATH = "label_encoder.pkl"

X_TEST_PATH = "X_test.pkl"
Y_TEST_PATH = "y_test.pkl"

SELECTED_MODEL_PATH = "selected_model.pkl"
MODEL_METADATA_PATH = "model_metadata.pkl"

MODEL_COMPARISON_PATH = "model_comparison.csv"
PERFORMANCE_GRAPH_PATH = "model_performance_comparison.png"


# ----------------------------------------------------------------
# 2. LOAD TEST DATA
# ----------------------------------------------------------------

print("=" * 70)
print("STEP 6: MODEL EVALUATION")
print("=" * 70)

print("\nLoading test data...")

X_test = joblib.load(X_TEST_PATH)
y_test = joblib.load(Y_TEST_PATH)

print("\nTest feature shape:")
print(X_test.shape)

print("\nTest target shape:")
print(y_test.shape)


# ----------------------------------------------------------------
# 3. LOAD LABEL ENCODER
# ----------------------------------------------------------------

print("\nLoading label encoder...")

label_encoder = joblib.load(
    ENCODER_PATH
)

class_names = label_encoder.classes_

print("\nDisease categories:")

for i, category in enumerate(class_names):
    print(f"{i}: {category}")

print("\nNumber of disease categories:")
print(len(class_names))


# ----------------------------------------------------------------
# 4. LOAD TRAINED MODELS
# ----------------------------------------------------------------

print("\nLoading trained models...")

logistic_model = joblib.load(
    LOGISTIC_PATH
)

nb_model = joblib.load(
    NB_PATH
)

svm_model = joblib.load(
    SVM_PATH
)

models = {
    "Logistic Regression": logistic_model,
    "Multinomial Naive Bayes": nb_model,
    "Linear SVM": svm_model
}

print("\nAll models loaded successfully.")


# ----------------------------------------------------------------
# 5. EVALUATE MODELS
# ----------------------------------------------------------------

results = []

for model_name, model in models.items():

    print("\n" + "=" * 70)
    print(f"EVALUATING: {model_name}")
    print("=" * 70)

    # ------------------------------------------------------------
    # Predictions
    # ------------------------------------------------------------

    y_pred = model.predict(
        X_test
    )

    # ------------------------------------------------------------
    # Accuracy
    # ------------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    # ------------------------------------------------------------
    # Weighted Metrics
    # ------------------------------------------------------------
    # Weighted metrics consider class size.
    # Larger disease categories have more influence.
    # ------------------------------------------------------------

    precision_weighted = precision_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    recall_weighted = recall_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    f1_weighted = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    # ------------------------------------------------------------
    # Macro Metrics
    # ------------------------------------------------------------
    # Macro metrics give equal importance to every disease category.
    # This is useful because the dataset is imbalanced.
    # ------------------------------------------------------------

    precision_macro = precision_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    recall_macro = recall_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    f1_macro = f1_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    # ------------------------------------------------------------
    # Display Metrics
    # ------------------------------------------------------------

    print("\nMODEL PERFORMANCE")

    print(f"\nAccuracy            : {accuracy:.4f}")

    print(
        f"Precision (Weighted): "
        f"{precision_weighted:.4f}"
    )

    print(
        f"Recall (Weighted)   : "
        f"{recall_weighted:.4f}"
    )

    print(
        f"F1 Score (Weighted) : "
        f"{f1_weighted:.4f}"
    )

    print(
        f"\nPrecision (Macro)   : "
        f"{precision_macro:.4f}"
    )

    print(
        f"Recall (Macro)      : "
        f"{recall_macro:.4f}"
    )

    print(
        f"F1 Score (Macro)    : "
        f"{f1_macro:.4f}"
    )


    # ------------------------------------------------------------
    # 6. CLASSIFICATION REPORT
    # ------------------------------------------------------------

    report = classification_report(
        y_test,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        zero_division=0
    )

    print("\nClassification Report:")
    print(report)

    # Safe filename
    safe_name = (
        model_name
        .lower()
        .replace(" ", "_")
    )

    report_path = (
        f"{safe_name}_classification_report.txt"
    )

    # Save classification report
    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            f"Model: {model_name}\n\n"
        )

        file.write(
            f"Accuracy: {accuracy:.4f}\n\n"
        )

        file.write(
            "WEIGHTED METRICS\n"
        )

        file.write(
            f"Precision Weighted: "
            f"{precision_weighted:.4f}\n"
        )

        file.write(
            f"Recall Weighted: "
            f"{recall_weighted:.4f}\n"
        )

        file.write(
            f"F1 Weighted: "
            f"{f1_weighted:.4f}\n\n"
        )

        file.write(
            "MACRO METRICS\n"
        )

        file.write(
            f"Precision Macro: "
            f"{precision_macro:.4f}\n"
        )

        file.write(
            f"Recall Macro: "
            f"{recall_macro:.4f}\n"
        )

        file.write(
            f"F1 Macro: "
            f"{f1_macro:.4f}\n\n"
        )

        file.write(
            "CLASSIFICATION REPORT\n"
        )

        file.write(report)

    print(
        f"\nClassification report saved: "
        f"{report_path}"
    )


    # ------------------------------------------------------------
    # 7. CONFUSION MATRIX
    # ------------------------------------------------------------

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=list(range(len(class_names)))
    )

    plt.figure(
        figsize=(12, 9)
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.title(
        f"Confusion Matrix - {model_name}"
    )

    plt.xlabel(
        "Predicted Disease Category"
    )

    plt.ylabel(
        "Actual Disease Category"
    )

    plt.xticks(
        rotation=45,
        ha="right"
    )

    plt.yticks(
        rotation=0
    )

    plt.tight_layout()

    confusion_path = (
        f"{safe_name}_confusion_matrix.png"
    )

    plt.savefig(
        confusion_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Confusion matrix saved: "
        f"{confusion_path}"
    )


    # ------------------------------------------------------------
    # 8. STORE MODEL RESULTS
    # ------------------------------------------------------------

    results.append({

        "model": model_name,

        "accuracy": accuracy,

        "precision_weighted":
            precision_weighted,

        "recall_weighted":
            recall_weighted,

        "f1_weighted":
            f1_weighted,

        "precision_macro":
            precision_macro,

        "recall_macro":
            recall_macro,

        "f1_macro":
            f1_macro
    })


# ----------------------------------------------------------------
# 9. CREATE MODEL COMPARISON TABLE
# ----------------------------------------------------------------

print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

results_df = pd.DataFrame(
    results
)


# ----------------------------------------------------------------
# 10. SORT MODELS USING MACRO F1 SCORE
# ----------------------------------------------------------------
# Macro F1 gives equal importance to all disease categories.
# This is useful for an imbalanced multiclass dataset.
# ----------------------------------------------------------------

results_df = results_df.sort_values(
    by="f1_macro",
    ascending=False
).reset_index(
    drop=True
)

print("\nModel Comparison:\n")

print(
    results_df.to_string(
        index=False
    )
)


# ----------------------------------------------------------------
# 11. SAVE MODEL COMPARISON
# ----------------------------------------------------------------

results_df.to_csv(
    MODEL_COMPARISON_PATH,
    index=False
)

print(
    f"\nModel comparison saved to: "
    f"{MODEL_COMPARISON_PATH}"
)


# ----------------------------------------------------------------
# 12. PERFORMANCE COMPARISON GRAPH
# ----------------------------------------------------------------

plot_df = results_df.set_index(
    "model"
)[
    [
        "accuracy",
        "precision_weighted",
        "recall_weighted",
        "f1_weighted",
        "precision_macro",
        "recall_macro",
        "f1_macro"
    ]
]


plt.figure(
    figsize=(14, 7)
)

plot_df.plot(
    kind="bar",
    ax=plt.gca()
)

plt.title(
    "Machine Learning Model Performance Comparison"
)

plt.xlabel(
    "Machine Learning Model"
)

plt.ylabel(
    "Performance Score"
)

plt.ylim(
    0,
    1
)

plt.xticks(
    rotation=20,
    ha="right"
)

plt.legend(
    title="Evaluation Metric",
    bbox_to_anchor=(1.05, 1),
    loc="upper left"
)

plt.tight_layout()

plt.savefig(
    PERFORMANCE_GRAPH_PATH,
    dpi=150,
    bbox_inches="tight"
)

plt.close()

print(
    f"\nPerformance comparison graph saved to: "
    f"{PERFORMANCE_GRAPH_PATH}"
)


# ----------------------------------------------------------------
# 13. SELECT FINAL MODEL
# ----------------------------------------------------------------
# Select model with highest Macro F1 Score
# ----------------------------------------------------------------

best_model_name = results_df.iloc[0][
    "model"
]

best_row = results_df.iloc[0]

print("\n" + "=" * 70)
print("SELECTED MODEL")
print("=" * 70)

print(
    "\nSelected model based on "
    "highest Macro F1-score:"
)

print(
    best_model_name
)

print(
    f"\nAccuracy: "
    f"{best_row['accuracy']:.4f}"
)

print(
    f"Weighted F1: "
    f"{best_row['f1_weighted']:.4f}"
)

print(
    f"Macro F1: "
    f"{best_row['f1_macro']:.4f}"
)


# ----------------------------------------------------------------
# 14. GET SELECTED MODEL OBJECT
# ----------------------------------------------------------------

best_model = models[
    best_model_name
]


# ----------------------------------------------------------------
# 15. SAVE SELECTED MODEL
# ----------------------------------------------------------------

joblib.dump(
    best_model,
    SELECTED_MODEL_PATH
)

print(
    f"\nSelected model saved to: "
    f"{SELECTED_MODEL_PATH}"
)


# ----------------------------------------------------------------
# 16. SAVE MODEL METADATA
# ----------------------------------------------------------------

model_metadata = {

    "selected_model":
        best_model_name,

    "accuracy":
        float(
            best_row["accuracy"]
        ),

    "precision_weighted":
        float(
            best_row[
                "precision_weighted"
            ]
        ),

    "recall_weighted":
        float(
            best_row[
                "recall_weighted"
            ]
        ),

    "f1_weighted":
        float(
            best_row[
                "f1_weighted"
            ]
        ),

    "precision_macro":
        float(
            best_row[
                "precision_macro"
            ]
        ),

    "recall_macro":
        float(
            best_row[
                "recall_macro"
            ]
        ),

    "f1_macro":
        float(
            best_row[
                "f1_macro"
            ]
        ),

    "classes":
        class_names.tolist(),

    "selection_metric":
        "Macro F1 Score"
}


joblib.dump(
    model_metadata,
    MODEL_METADATA_PATH
)

print(
    f"Model metadata saved to: "
    f"{MODEL_METADATA_PATH}"
)


# ----------------------------------------------------------------
# 17. FINAL SUMMARY
# ----------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 6 COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nGenerated Classification Reports:")

print(
    "  - logistic_regression_classification_report.txt"
)

print(
    "  - multinomial_naive_bayes_classification_report.txt"
)

print(
    "  - linear_svm_classification_report.txt"
)


print("\nGenerated Confusion Matrices:")

print(
    "  - logistic_regression_confusion_matrix.png"
)

print(
    "  - multinomial_naive_bayes_confusion_matrix.png"
)

print(
    "  - linear_svm_confusion_matrix.png"
)


print("\nGenerated Comparison Files:")

print(
    "  - model_comparison.csv"
)

print(
    "  - model_performance_comparison.png"
)


print("\nFinal Model Files:")

print(
    "  - selected_model.pkl"
)

print(
    "  - model_metadata.pkl"
)


print("\nSelected Model:")
print(
    best_model_name
)

print(
    "\nSelection Metric:"
)

print(
    "Macro F1 Score"
)

print(
    "\nEvaluation completed successfully."
)
