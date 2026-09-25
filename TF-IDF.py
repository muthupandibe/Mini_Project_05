# ================================================================
# STEP 4: TF-IDF FEATURE EXTRACTION
# Project: Clinical Trial Disease Category Classification
# ================================================================

import os
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer

# ----------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------

INPUT_PATH = "clinical_trials_cleaned.csv"
VECTORIZER_PATH = "tfidf_vectorizer.pkl"

TOP_FEATURES_PATH = "top_tfidf_features.csv"

# ----------------------------------------------------------------
# 1. LOAD CLEANED DATA
# ----------------------------------------------------------------

print("=" * 70)
print("STEP 4: TF-IDF FEATURE EXTRACTION")
print("=" * 70)

df = pd.read_csv(INPUT_PATH)

print("\nDataset shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

# ----------------------------------------------------------------
# 2. CHECK REQUIRED COLUMN
# ----------------------------------------------------------------

if "cleaned_summary" not in df.columns:
    raise ValueError(
        "Column 'cleaned_summary' was not found in "
        "clinical_trials_cleaned.csv"
    )

# Remove missing text safely
df["cleaned_summary"] = (
    df["cleaned_summary"]
    .fillna("")
    .astype(str)
)

# Remove empty summaries
df = df[df["cleaned_summary"].str.strip() != ""].copy()

print("\nRecords available for TF-IDF:")
print(len(df))

# ----------------------------------------------------------------
# 3. CREATE TF-IDF VECTORIZER
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
# 4. FIT AND TRANSFORM TEXT
# ----------------------------------------------------------------

X = vectorizer.fit_transform(df["cleaned_summary"])

print("\nTF-IDF transformation completed.")

print("\nTF-IDF matrix shape:")
print(X.shape)

print("\nNumber of documents:")
print(X.shape[0])

print("\nNumber of TF-IDF features:")
print(X.shape[1])

# ----------------------------------------------------------------
# 5. SAVE TF-IDF VECTORIZER
# ----------------------------------------------------------------

joblib.dump(
    vectorizer,
    VECTORIZER_PATH
)

print(f"\nTF-IDF vectorizer saved to: {VECTORIZER_PATH}")

# ----------------------------------------------------------------
# 6. GET FEATURE NAMES
# ----------------------------------------------------------------

feature_names = vectorizer.get_feature_names_out()

print("\nFirst 50 TF-IDF features:")

for feature in feature_names[:50]:
    print(feature)

# ----------------------------------------------------------------
# 7. IDENTIFY IMPORTANT TERMS
# ----------------------------------------------------------------

# Mean TF-IDF value for each feature
mean_tfidf = X.mean(axis=0).A1

tfidf_features = pd.DataFrame({
    "term": feature_names,
    "mean_tfidf": mean_tfidf
})

tfidf_features = (
    tfidf_features
    .sort_values(
        "mean_tfidf",
        ascending=False
    )
    .reset_index(drop=True)
)

# Save top TF-IDF features
tfidf_features.head(100).to_csv(
    TOP_FEATURES_PATH,
    index=False
)

print(
    f"\nTop TF-IDF features saved to: "
    f"{TOP_FEATURES_PATH}"
)

print("\nTop 20 TF-IDF features:")
print(
    tfidf_features.head(20).to_string(index=False)
)

# ----------------------------------------------------------------
# 8. SPARSE MATRIX INFORMATION
# ----------------------------------------------------------------

print("\n" + "=" * 70)
print("SPARSE MATRIX INFORMATION")
print("=" * 70)

print("\nMatrix shape:")
print(X.shape)

print("\nNumber of non-zero values:")
print(X.nnz)

print("\nSparsity:")
sparsity = 1 - (X.nnz / (X.shape[0] * X.shape[1]))

print(f"{sparsity:.4%}")

# ----------------------------------------------------------------
# 9. SAMPLE TF-IDF VALUES
# ----------------------------------------------------------------

print("\n" + "=" * 70)
print("SAMPLE TF-IDF VALUES")
print("=" * 70)

sample_features = feature_names[:10]

sample_matrix = X[:3, :10].toarray()

sample_df = pd.DataFrame(
    sample_matrix,
    columns=sample_features
)

print(sample_df)

# ----------------------------------------------------------------
# 10. FINAL OUTPUT
# ----------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 4 COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nFiles created:")

print("  - tfidf_vectorizer.pkl")
print("  - top_tfidf_features.csv")

print("\nTF-IDF is ready for machine-learning model training.")
