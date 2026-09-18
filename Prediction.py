# ================================================================
# STEP 7: DISEASE CATEGORY PREDICTION
# Project: Clinical Trial Disease Category Classification
# ================================================================

import re
import joblib
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ----------------------------------------------------------------
# 1. DOWNLOAD / LOAD NLTK RESOURCES
# ----------------------------------------------------------------

nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

# ----------------------------------------------------------------
# 2. LOAD TRAINED COMPONENTS
# ----------------------------------------------------------------

VECTORIZER_PATH = "tfidf_vectorizer.pkl"
MODEL_PATH = "selected_model.pkl"
ENCODER_PATH = "label_encoder.pkl"
METADATA_PATH = "model_metadata.pkl"

print("=" * 70)
print("STEP 7: DISEASE CATEGORY PREDICTION")
print("=" * 70)

print("\nLoading trained components...")

vectorizer = joblib.load(
    VECTORIZER_PATH
)

model = joblib.load(
    MODEL_PATH
)

label_encoder = joblib.load(
    ENCODER_PATH
)

model_metadata = joblib.load(
    METADATA_PATH
)

print("TF-IDF vectorizer loaded.")
print("Selected model loaded.")
print("Label encoder loaded.")
print("Model metadata loaded.")

print(
    "\nSelected model:",
    model_metadata["selected_model"]
)

# ----------------------------------------------------------------
# 3. TEXT PREPROCESSING
# ----------------------------------------------------------------

stop_words = set(
    stopwords.words("english")
)

# Keep medically meaningful negative words
stop_words = stop_words - {
    "no",
    "not",
    "nor"
}

lemmatizer = WordNetLemmatizer()


def clean_medical_text(text):
    """
    Apply the same preprocessing used during model training.
    """

    text = str(text)

    # Lowercase
    text = text.lower()

    # Remove HTML
    text = re.sub(
        r"<.*?>",
        " ",
        text
    )

    # Remove URLs
    text = re.sub(
        r"http\S+|www\S+",
        " ",
        text
    )

    # Remove special characters
    text = re.sub(
        r"[^a-zA-Z0-9\s]",
        " ",
        text
    )

    # Remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    # Tokenization
    tokens = text.split()

    # Stopword removal
    tokens = [
        word
        for word in tokens
        if word not in stop_words
    ]

    # Lemmatization
    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
    ]

    return " ".join(tokens)


# ----------------------------------------------------------------
# 4. PREDICTION FUNCTION
# ----------------------------------------------------------------

def predict_disease_category(text):

    if not text or not text.strip():
        raise ValueError(
            "Clinical trial summary cannot be empty."
        )

    # Preprocess text
    cleaned_text = clean_medical_text(text)

    if not cleaned_text.strip():
        raise ValueError(
            "The clinical trial summary contains no usable text "
            "after preprocessing."
        )

    # Convert text to TF-IDF
    text_vector = vectorizer.transform(
        [cleaned_text]
    )

    # Predict encoded category
    prediction_encoded = model.predict(
        text_vector
    )[0]

    # Convert encoded value back to category
    prediction = label_encoder.inverse_transform(
        [prediction_encoded]
    )[0]

    return prediction, cleaned_text


# ----------------------------------------------------------------
# 5. TEST SAMPLE
# ----------------------------------------------------------------

sample_summary = """
This randomized clinical trial evaluates the safety and efficacy
of a new treatment in adult patients with heart failure and
cardiovascular disease. The study measures cardiac function,
hospitalization, mortality, and other cardiovascular outcomes.
"""

print("\n" + "=" * 70)
print("SAMPLE CLINICAL TRIAL")
print("=" * 70)

print("\nOriginal Summary:")
print(sample_summary)

# Predict
prediction, cleaned_text = predict_disease_category(
    sample_summary
)

print("\nCleaned Summary:")
print(cleaned_text)

print("\n" + "=" * 70)
print("PREDICTION RESULT")
print("=" * 70)

print("\nPredicted Disease Category:")
print(prediction)

print(
    "\nModel used:",
    model_metadata["selected_model"]
)

# ----------------------------------------------------------------
# 6. PROBABILITY / DECISION INFORMATION
# ----------------------------------------------------------------

print("\n" + "=" * 70)
print("MODEL CONFIDENCE INFORMATION")
print("=" * 70)

text_vector = vectorizer.transform(
    [cleaned_text]
)

# Models such as Logistic Regression and Naive Bayes
# provide predict_proba().
if hasattr(model, "predict_proba"):

    probabilities = model.predict_proba(
        text_vector
    )[0]

    probability_df = []

    for category, probability in zip(
        label_encoder.classes_,
        probabilities
    ):

        probability_df.append({
            "disease_category": category,
            "probability": probability
        })

    probability_df = sorted(
        probability_df,
        key=lambda x: x["probability"],
        reverse=True
    )

    print("\nCategory probabilities:")

    for item in probability_df:
        print(
            f"{item['disease_category']}: "
            f"{item['probability']:.4f}"
        )

else:

    print(
        "\nThe selected model does not provide "
        "probability estimates."
    )

# ----------------------------------------------------------------
# 7. FINAL MESSAGE
# ----------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 7 COMPLETED SUCCESSFULLY")
print("=" * 70)

print(
    "\nThe trained pipeline can now classify "
    "new clinical trial summaries."
)


# ================================================================
# SHARED TEXT PREPROCESSING
# ================================================================

import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# Download required NLTK resources
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")


# Stopwords
stop_words = set(
    stopwords.words("english")
)

# Preserve medically meaningful negative words
stop_words = stop_words - {
    "no",
    "not",
    "nor"
}

# Lemmatizer
lemmatizer = WordNetLemmatizer()


def clean_medical_text(text):
    """
    Clean clinical trial text using the same
    preprocessing logic used during training.
    """

    text = str(text)

    # Lowercase
    text = text.lower()

    # Remove HTML
    text = re.sub(
        r"<.*?>",
        " ",
        text
    )

    # Remove URLs
    text = re.sub(
        r"http\S+|www\S+",
        " ",
        text
    )

    # Remove special characters
    text = re.sub(
        r"[^a-zA-Z0-9\s]",
        " ",
        text
    )

    # Remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    # Tokenization
    tokens = text.split()

    # Stopword removal
    tokens = [
        word
        for word in tokens
        if word not in stop_words
    ]

    # Lemmatization
    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
    ]

    return " ".join(tokens)