# ================================================================
# STEP 5: MACHINE LEARNING MODEL TRAINING
# Project: Clinical Trial Disease Category Classification
# ================================================================

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC


# ----------------------------------------------------------------
# 1. CONFIGURATION
# ----------------------------------------------------------------

DATA_PATH = "clinical_trials_cleaned.csv"

# Final TF-IDF vectorizer used by trained models and Streamlit
VECTORIZER_PATH = "tfidf_vectorizer.pkl"

# Trained model files
LOGISTIC_PATH = "logistic_regression.pkl"
NB_PATH = "multinomial_naive_bayes.pkl"
SVM_PATH = "linear_svm.pkl"

# Label encoder
ENCODER_PATH = "label_encoder.pkl"

# Test data for Step 6 evaluation
X_TEST_PATH = "X_test.pkl"
Y_TEST_PATH = "y_test.pkl"


# ----------------------------------------------------------------
# 2. LOAD CLEANED DATA
# ----------------------------------------------------------------

print("=" * 70)
print("STEP 5: MACHINE LEARNING MODEL TRAINING")
print("=" * 70)

df = pd.read_csv(DATA_PATH)

print("\nDataset shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())


# ----------------------------------------------------------------
# 3. CHECK REQUIRED COLUMNS
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
# 4. CLEAN INPUT DATA
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

# Remove empty text and empty target records
df = df[
    (df["cleaned_summary"].str.strip() != "") &
    (df["disease_category"].str.strip() != "")
].copy()

df = df.reset_index(drop=True)

print("\nRecords available for model training:")
print(len(df))


# ----------------------------------------------------------------
# 5. PREPARE TEXT INPUT AND TARGET
# ----------------------------------------------------------------

# Input text
X_text = df["cleaned_summary"]

# Target disease category
y_text = df["disease_category"]

print("\nInput text records:")
print(len(X_text))

print("\nDisease category distribution:")
print(y_text.value_counts())


# ----------------------------------------------------------------
# 6. ENCODE DISEASE CATEGORIES
# ----------------------------------------------------------------

print("\nEncoding disease categories...")

label_encoder = LabelEncoder()

y = label_encoder.fit_transform(y_text)

print("\nDisease categories:")

for index, category in enumerate(label_encoder.classes_):

    print(
        f"{index}: {category}"
    )

print("\nNumber of disease categories:")
print(len(label_encoder.classes_))


# ----------------------------------------------------------------
# 7. TRAIN / TEST SPLIT
# ----------------------------------------------------------------
# IMPORTANT:
# Split the TEXT before fitting TF-IDF.
#
# This prevents information from the test dataset from being used
# when TF-IDF learns vocabulary and IDF values.
# ----------------------------------------------------------------

print("\nCreating stratified train-test split...")

X_train_text, X_test_text, y_train, y_test = train_test_split(
    X_text,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:")
print(len(X_train_text))

print("\nTesting samples:")
print(len(X_test_text))


# ----------------------------------------------------------------
# 8. CREATE TF-IDF VECTORIZER
# ----------------------------------------------------------------

print("\nCreating TF-IDF vectorizer...")

vectorizer = TfidfVectorizer(
    max_features=20000,
    min_df=2,
    max_df=0.95,
    ngram_range=(1, 2),
    sublinear_tf=True,
    strip_accents="unicode"
)


# ----------------------------------------------------------------
# 9. FIT TF-IDF ONLY ON TRAINING DATA
# ----------------------------------------------------------------

print("\nFitting TF-IDF on training data...")

X_train = vectorizer.fit_transform(
    X_train_text
)

print("\nTransforming test data...")

X_test = vectorizer.transform(
    X_test_text
)

print("\nTF-IDF transformation completed.")

print("\nTraining TF-IDF matrix shape:")
print(X_train.shape)

print("\nTesting TF-IDF matrix shape:")
print(X_test.shape)

print("\nNumber of TF-IDF features:")
print(len(vectorizer.get_feature_names_out()))


# ----------------------------------------------------------------
# 10. SAVE FINAL TF-IDF VECTORIZER
# ----------------------------------------------------------------
# This is the vectorizer that must also be used by Streamlit.
# ----------------------------------------------------------------

joblib.dump(
    vectorizer,
    VECTORIZER_PATH
)

print(
    f"\nFinal training TF-IDF vectorizer saved to: "
    f"{VECTORIZER_PATH}"
)


# ----------------------------------------------------------------
# 11. TRAIN LOGISTIC REGRESSION
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

print(
    "Logistic Regression training completed."
)


# ----------------------------------------------------------------
# 12. TRAIN MULTINOMIAL NAIVE BAYES
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

print(
    "Multinomial Naive Bayes training completed."
)


# ----------------------------------------------------------------
# 13. TRAIN LINEAR SVM
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

print(
    "Linear SVM training completed."
)


# ----------------------------------------------------------------
# 14. SAVE TRAINED MODELS
# ----------------------------------------------------------------

print("\n" + "=" * 70)
print("SAVING TRAINED MODELS")
print("=" * 70)

joblib.dump(
    logistic_model,
    LOGISTIC_PATH
)

print(
    f"Saved: {LOGISTIC_PATH}"
)

joblib.dump(
    nb_model,
    NB_PATH
)

print(
    f"Saved: {NB_PATH}"
)

joblib.dump(
    svm_model,
    SVM_PATH
)

print(
    f"Saved: {SVM_PATH}"
)


# ----------------------------------------------------------------
# 15. SAVE LABEL ENCODER
# ----------------------------------------------------------------

joblib.dump(
    label_encoder,
    ENCODER_PATH
)

print(
    f"Saved: {ENCODER_PATH}"
)


# ----------------------------------------------------------------
# 16. SAVE TEST DATA FOR STEP 6
# ----------------------------------------------------------------

joblib.dump(
    X_test,
    X_TEST_PATH
)

joblib.dump(
    y_test,
    Y_TEST_PATH
)

print(
    f"Saved: {X_TEST_PATH}"
)

print(
    f"Saved: {Y_TEST_PATH}"
)


# ----------------------------------------------------------------
# 17. MODEL INFORMATION
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
# 18. TRAINING SUMMARY
# ----------------------------------------------------------------

print("\n" + "=" * 70)
print("TRAINING SUMMARY")
print("=" * 70)

print(
    f"\nTotal records: {len(df)}"
)

print(
    f"Training records: {len(X_train_text)}"
)

print(
    f"Testing records: {len(X_test_text)}"
)

print(
    f"Disease categories: "
    f"{len(label_encoder.classes_)}"
)

print(
    f"TF-IDF features: "
    f"{X_train.shape[1]}"
)


# ----------------------------------------------------------------
# 19. FINAL OUTPUT
# ----------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 5 COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nGenerated files:")

print("  - tfidf_vectorizer.pkl")
print("  - logistic_regression.pkl")
print("  - multinomial_naive_bayes.pkl")
print("  - linear_svm.pkl")
print("  - label_encoder.pkl")
print("  - X_test.pkl")
print("  - y_test.pkl")

print(
    "\nThree machine-learning models "
    "are ready for evaluation."
)
