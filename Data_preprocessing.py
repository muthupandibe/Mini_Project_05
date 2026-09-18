# ================================================================
# STEP 2: DATA PREPROCESSING
# ================================================================

import pandas as pd
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ----------------------------------------------------------------
# 1. Download required NLTK resources
# ----------------------------------------------------------------

nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

# ----------------------------------------------------------------
# 2. File paths
# ----------------------------------------------------------------

INPUT_PATH = "clinical_trials_selected.csv"
OUTPUT_PATH = "clinical_trials_cleaned.csv"

# ----------------------------------------------------------------
# 3. Load selected dataset
# ----------------------------------------------------------------

df = pd.read_csv(INPUT_PATH)

print("Original shape:", df.shape)

# ----------------------------------------------------------------
# 4. Check missing values
# ----------------------------------------------------------------

print("\nMissing values before preprocessing:")
print(df.isna().sum())

# ----------------------------------------------------------------
# 5. Remove missing values
# ----------------------------------------------------------------

df["brief_summary"] = df["brief_summary"].fillna("")
df["disease_category"] = df["disease_category"].fillna("")

# Remove rows where text or target is empty
df = df[
    (df["brief_summary"].str.strip() != "") &
    (df["disease_category"].str.strip() != "")
].copy()

# ----------------------------------------------------------------
# 6. Remove duplicate records
# ----------------------------------------------------------------

duplicates_before = df.duplicated(subset=["nct_id"]).sum()

print("\nDuplicate nct_id records:", duplicates_before)

df = df.drop_duplicates(subset=["nct_id"])

# ----------------------------------------------------------------
# 7. Text preprocessing function
# ----------------------------------------------------------------

stop_words = set(stopwords.words("english"))

# Keep medically useful negation words
stop_words = stop_words - {"no", "not", "nor"}

lemmatizer = WordNetLemmatizer()


def clean_medical_text(text):

    # Convert to string
    text = str(text)

    # Convert to lowercase
    text = text.lower()

    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)

    # Remove punctuation and special characters
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Tokenization
    tokens = text.split()

    # Stop-word removal
    tokens = [
        word for word in tokens
        if word not in stop_words
    ]

    # Lemmatization
    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
    ]

    return " ".join(tokens)


# ----------------------------------------------------------------
# 8. Apply text preprocessing
# ----------------------------------------------------------------

print("\nApplying medical text preprocessing...")

df["cleaned_summary"] = df["brief_summary"].apply(
    clean_medical_text
)

# ----------------------------------------------------------------
# 9. Remove empty cleaned text
# ----------------------------------------------------------------

df = df[
    df["cleaned_summary"].str.strip() != ""
].copy()

# ----------------------------------------------------------------
# 10. Text statistics
# ----------------------------------------------------------------

df["original_text_length"] = df["brief_summary"].str.len()

df["cleaned_text_length"] = df["cleaned_summary"].str.len()

df["word_count"] = df["cleaned_summary"].str.split().str.len()

# ----------------------------------------------------------------
# 11. Check final dataset
# ----------------------------------------------------------------

print("\nFinal shape:", df.shape)

print("\nFinal missing values:")
print(df.isna().sum())

print("\nSample cleaned text:")

print(df[
    [
        "brief_summary",
        "cleaned_summary",
        "disease_category"
    ]
].head())

# ----------------------------------------------------------------
# 12. Save cleaned dataset
# ----------------------------------------------------------------

df.to_csv(
    OUTPUT_PATH,
    index=False
)

print(
    f"\nCleaned dataset saved to: {OUTPUT_PATH}"
)