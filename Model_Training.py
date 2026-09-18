# ================================================================
# STEP 5: MACHINE LEARNING MODEL TRAINING
# Project: Clinical Trial Disease Category Classification
# ================================================================

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

# ----------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------

DATA_PATH = "clinical_trials_cleaned.csv"
VECTORIZER_PATH = "tfidf_vectorizer.pkl"

LOGISTIC_PATH = "logistic_regression.pkl"
NB_PATH = "multinomial_naive_bayes.pkl"
SVM_PATH = "linear_svm.pkl"

ENCODER_PATH = "label_encoder.pkl"

X_TEST_PATH = "X_test.pkl"
Y_TEST_PATH = "y_test.pkl"

# ----------------------------------------------------------------
# 1. LOAD DATA
# ----------------------------------------------------------------

print("=" * 70)
print("STEP 5: MACHINE LEARNING MODEL TRAINING")
print("=" * 70)

df = pd.read_csv(DATA_PATH)

print("\nDataset shape:")
print(df.shape)

# ----------------------------------------------------------------
# 2. CHECK REQUIRED COLUMNS
# ----------------------------------------------------------------

required_columns = [
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
# 3. CLEAN INPUT DATA
# ----------------------------------------------------------------

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

# Remove empty records
df = df[
    (df["cleaned_summary"].str.strip() != "") &
    (df["disease_category"].str.strip() != "")
].copy()

df = df.reset_index(drop=True)

print("\nRecords used for training:")
print(len(df))

# ----------------------------------------------------------------
# 4. LOAD TF-IDF VECTORIZER
# ----------------------------------------------------------------

print("\nLoading TF-IDF vectorizer...")

vectorizer = joblib.load(VECTORIZER_PATH)

# Transform cleaned text
X = vectorizer.transform(
    df["cleaned_summary"]
)

print("\nTF-IDF matrix shape:")
print(X.shape)

# ----------------------------------------------------------------
# 5. ENCODE DISEASE CATEGORIES
# ----------------------------------------------------------------

print("\nEncoding disease categories...")

label_encoder = LabelEncoder()

y = label_encoder.fit_transform(
    df["disease_category"]
)

print("\nDisease categories:")

for index, category in enumerate(label_encoder.classes_):
    print(f"{index}: {category}")

print("\nNumber of disease categories:")
print(len(label_encoder.classes_))

# ----------------------------------------------------------------
# 6. TRAIN / TEST SPLIT
# ----------------------------------------------------------------

print("\nCreating train-test split...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:")
print(X_train.shape[0])

print("Testing samples:")
print(X_test.shape[0])

# ----------------------------------------------------------------
# 7. TRAIN LOGISTIC REGRESSION
# ----------------------------------------------------------------

print("\n" + "=" * 70)
print("MODEL 1: LOGISTIC REGRESSION")
print("=" * 70)

logistic_model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    random_state=42
)

logistic_model.fit(
    X_train,
    y_train
)

print("Logistic Regression training completed.")

# ----------------------------------------------------------------
# 8. TRAIN MULTINOMIAL NAIVE BAYES
# ----------------------------------------------------------------

print("\n" + "=" * 70)
print("MODEL 2: MULTINOMIAL NAIVE BAYES")
print("=" * 70)

nb_model = MultinomialNB(
    alpha=0.5
)

nb_model.fit(
    X_train,
    y_train
)

print("Multinomial Naive Bayes training completed.")

# ----------------------------------------------------------------
# 9. TRAIN LINEAR SVM
# ----------------------------------------------------------------

print("\n" + "=" * 70)
print("MODEL 3: LINEAR SVM")
print("=" * 70)

svm_model = LinearSVC(
    class_weight="balanced",
    random_state=42
)

svm_model.fit(
    X_train,
    y_train
)

print("Linear SVM training completed.")

# ----------------------------------------------------------------
# 10. SAVE TRAINED MODELS
# ----------------------------------------------------------------

print("\n" + "=" * 70)
print("SAVING TRAINED MODELS")
print("=" * 70)

joblib.dump(
    logistic_model,
    LOGISTIC_PATH
)

print(f"Saved: {LOGISTIC_PATH}")

joblib.dump(
    nb_model,
    NB_PATH
)

print(f"Saved: {NB_PATH}")

joblib.dump(
    svm_model,
    SVM_PATH
)

print(f"Saved: {SVM_PATH}")

# ----------------------------------------------------------------
# 11. SAVE LABEL ENCODER
# ----------------------------------------------------------------

joblib.dump(
    label_encoder,
    ENCODER_PATH
)

print(f"Saved: {ENCODER_PATH}")

# ----------------------------------------------------------------
# 12. SAVE TEST DATA
# ----------------------------------------------------------------

joblib.dump(
    X_test,
    X_TEST_PATH
)

joblib.dump(
    y_test,
    Y_TEST_PATH
)

print(f"Saved: {X_TEST_PATH}")
print(f"Saved: {Y_TEST_PATH}")

# ----------------------------------------------------------------
# 13. MODEL INFORMATION
# ----------------------------------------------------------------

print("\n" + "=" * 70)
print("MODEL INFORMATION")
print("=" * 70)

print("\nLogistic Regression:")
print(logistic_model)

print("\nMultinomial Naive Bayes:")
print(nb_model)

print("\nLinear SVM:")
print(svm_model)

# ----------------------------------------------------------------
# 14. FINAL OUTPUT
# ----------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 5 COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nGenerated files:")

print("  - logistic_regression.pkl")
print("  - multinomial_naive_bayes.pkl")
print("  - linear_svm.pkl")
print("  - label_encoder.pkl")
print("  - X_test.pkl")
print("  - y_test.pkl")

print("\nThree machine-learning models are ready for evaluation.")