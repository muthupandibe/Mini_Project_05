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
# Configuration
# ----------------------------------------------------------------

LOGISTIC_PATH = "logistic_regression.pkl"
NB_PATH = "multinomial_naive_bayes.pkl"
SVM_PATH = "linear_svm.pkl"

ENCODER_PATH = "label_encoder.pkl"

X_TEST_PATH = "X_test.pkl"
Y_TEST_PATH = "y_test.pkl"

# ----------------------------------------------------------------
# 1. LOAD TEST DATA
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
# 2. LOAD LABEL ENCODER
# ----------------------------------------------------------------

label_encoder = joblib.load(
    ENCODER_PATH
)

class_names = label_encoder.classes_

print("\nDisease categories:")

for i, category in enumerate(class_names):
    print(f"{i}: {category}")

# ----------------------------------------------------------------
# 3. LOAD MODELS
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

print("All models loaded successfully.")

# ----------------------------------------------------------------
# 4. EVALUATE MODELS
# ----------------------------------------------------------------

results = []

for model_name, model in models.items():

    print("\n" + "=" * 70)
    print(f"EVALUATING: {model_name}")
    print("=" * 70)

    # Predictions
    y_pred = model.predict(X_test)

    # Metrics
    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    print(f"\nAccuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    # ------------------------------------------------------------
    # Classification Report
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

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            f"Model: {model_name}\n\n"
        )

        file.write(
            f"Accuracy : {accuracy:.4f}\n"
        )

        file.write(
            f"Precision: {precision:.4f}\n"
        )

        file.write(
            f"Recall   : {recall:.4f}\n"
        )

        file.write(
            f"F1 Score : {f1:.4f}\n\n"
        )

        file.write(
            "Classification Report:\n"
        )

        file.write(report)

    # ------------------------------------------------------------
    # Confusion Matrix
    # ------------------------------------------------------------

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=list(range(len(class_names)))
    )

    plt.figure(
        figsize=(10, 8)
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

    # Store metrics
    results.append({
        "model": model_name,
        "accuracy": accuracy,
        "precision_weighted": precision,
        "recall_weighted": recall,
        "f1_weighted": f1
    })

# ----------------------------------------------------------------
# 5. MODEL COMPARISON
# ----------------------------------------------------------------

print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "f1_weighted",
    ascending=False
).reset_index(drop=True)

print("\n")
print(
    results_df.to_string(
        index=False
    )
)

# Save comparison
results_df.to_csv(
    "model_comparison.csv",
    index=False
)

print(
    "\nModel comparison saved to: "
    "model_comparison.csv"
)

# ----------------------------------------------------------------
# 6. PERFORMANCE COMPARISON GRAPH
# ----------------------------------------------------------------

plot_df = results_df.set_index(
    "model"
)[
    [
        "accuracy",
        "precision_weighted",
        "recall_weighted",
        "f1_weighted"
    ]
]

plt.figure(
    figsize=(12, 6)
)

plot_df.plot(
    kind="bar",
    ax=plt.gca()
)

plt.title(
    "Machine Learning Model Performance Comparison"
)

plt.xlabel(
    "Model"
)

plt.ylabel(
    "Score"
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
    title="Metric"
)

plt.tight_layout()

plt.savefig(
    "model_performance_comparison.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close()

# ----------------------------------------------------------------
# 7. SELECT MODEL
# ----------------------------------------------------------------

best_model_name = results_df.iloc[0]["model"]

print("\n" + "=" * 70)
print("SELECTED MODEL")
print("=" * 70)

print(
    f"\nSelected model based on highest weighted F1-score:"
)

print(
    best_model_name
)

# Get actual model object
best_model = models[
    best_model_name
]

# ----------------------------------------------------------------
# 8. SAVE SELECTED MODEL
# ----------------------------------------------------------------

joblib.dump(
    best_model,
    "selected_model.pkl"
)

print(
    "\nSelected model saved to: "
    "selected_model.pkl"
)

# ----------------------------------------------------------------
# 9. SAVE MODEL METADATA
# ----------------------------------------------------------------

best_row = results_df.iloc[0]

model_metadata = {
    "selected_model": best_model_name,
    "accuracy": float(best_row["accuracy"]),
    "precision_weighted": float(
        best_row["precision_weighted"]
    ),
    "recall_weighted": float(
        best_row["recall_weighted"]
    ),
    "f1_weighted": float(
        best_row["f1_weighted"]
    ),
    "classes": class_names.tolist()
}

joblib.dump(
    model_metadata,
    "model_metadata.pkl"
)

print(
    "Model metadata saved to: "
    "model_metadata.pkl"
)

# ----------------------------------------------------------------
# 10. FINAL SUMMARY
# ----------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 6 COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nGenerated files:")

print("  - logistic_regression_classification_report.txt")
print("  - multinomial_naive_bayes_classification_report.txt")
print("  - linear_svm_classification_report.txt")

print("  - logistic_regression_confusion_matrix.png")
print("  - multinomial_naive_bayes_confusion_matrix.png")
print("  - linear_svm_confusion_matrix.png")

print("  - model_comparison.csv")
print("  - model_performance_comparison.png")
print("  - selected_model.pkl")
print("  - model_metadata.pkl")

print("\nSelected model:")
print(best_model_name)

print("\nEvaluation completed successfully.")