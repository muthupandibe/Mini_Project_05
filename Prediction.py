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
# 1. NLTK RESOURCES
# ----------------------------------------------------------------

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)


# ----------------------------------------------------------------
# 2. FILE PATHS
# ----------------------------------------------------------------

VECTORIZER_PATH = "tfidf_vectorizer.pkl"
MODEL_PATH = "selected_model.pkl"
ENCODER_PATH = "label_encoder.pkl"
METADATA_PATH = "model_metadata.pkl"


# ----------------------------------------------------------------
# 3. LOAD TRAINED COMPONENTS
# ----------------------------------------------------------------

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


print("TF-IDF vectorizer loaded successfully.")
print("Selected model loaded successfully.")
print("Label encoder loaded successfully.")
print("Model metadata loaded successfully.")


print(
    "\nSelected Model:",
    model_metadata["selected_model"]
)

print(
    "Model Selection Metric:",
    model_metadata.get(
        "selection_metric",
        "Not available"
    )
)


# ----------------------------------------------------------------
# 4. DISPLAY AVAILABLE DISEASE CATEGORIES
# ----------------------------------------------------------------

print("\nDisease categories supported by the model:")

for category in label_encoder.classes_:
    print(f"  - {category}")


# ----------------------------------------------------------------
# 5. TEXT PREPROCESSING SETUP
# ----------------------------------------------------------------

stop_words = set(
    stopwords.words("english")
)

# Keep medically meaningful negation words
stop_words = stop_words - {
    "no",
    "not",
    "nor"
}

lemmatizer = WordNetLemmatizer()


# ----------------------------------------------------------------
# 6. TEXT PREPROCESSING FUNCTION
# ----------------------------------------------------------------

def clean_medical_text(text):
    """
    Apply the same preprocessing used during
    model training.
    """

    # Convert to string
    text = str(text)

    # Convert to lowercase
    text = text.lower()

    # Remove HTML tags
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

    # Remove punctuation and special characters
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
# 7. DISEASE PREDICTION FUNCTION
# ----------------------------------------------------------------

def predict_disease_category(text):
    """
    Predict the disease category for a
    clinical trial brief summary.
    """

    # Check empty input
    if not text or not str(text).strip():

        raise ValueError(
            "Clinical trial summary cannot be empty."
        )

    # ------------------------------------------------------------
    # Preprocess input
    # ------------------------------------------------------------

    cleaned_text = clean_medical_text(
        text
    )

    if not cleaned_text.strip():

        raise ValueError(
            "The clinical trial summary contains "
            "no usable text after preprocessing."
        )

    # ------------------------------------------------------------
    # TF-IDF transformation
    # ------------------------------------------------------------

    text_vector = vectorizer.transform(
        [cleaned_text]
    )

    # ------------------------------------------------------------
    # Model prediction
    # ------------------------------------------------------------

    encoded_prediction = model.predict(
        text_vector
    )[0]

    # ------------------------------------------------------------
    # Decode disease category
    # ------------------------------------------------------------

    predicted_category = (
        label_encoder.inverse_transform(
            [encoded_prediction]
        )[0]
    )

    return {
        "original_text": text,
        "cleaned_text": cleaned_text,
        "encoded_prediction": int(
            encoded_prediction
        ),
        "predicted_category":
            predicted_category,
        "text_vector":
            text_vector
    }


# ----------------------------------------------------------------
# 8. MODEL CONFIDENCE FUNCTION
# ----------------------------------------------------------------

def get_prediction_scores(text_vector):
    """
    Return available prediction scores.

    Logistic Regression / Naive Bayes:
        predict_proba()

    Linear SVM:
        decision_function()
    """

    # ------------------------------------------------------------
    # Probability-based models
    # ------------------------------------------------------------

    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(
            text_vector
        )[0]

        results = []

        for category, probability in zip(
            label_encoder.classes_,
            probabilities
        ):

            results.append({
                "disease_category":
                    category,

                "score":
                    float(probability),

                "score_type":
                    "probability"
            })

        results = sorted(
            results,
            key=lambda x: x["score"],
            reverse=True
        )

        return results


    # ------------------------------------------------------------
    # Linear SVM
    # ------------------------------------------------------------

    elif hasattr(
        model,
        "decision_function"
    ):

        scores = model.decision_function(
            text_vector
        )[0]

        results = []

        for category, score in zip(
            label_encoder.classes_,
            scores
        ):

            results.append({
                "disease_category":
                    category,

                "score":
                    float(score),

                "score_type":
                    "decision_score"
            })

        results = sorted(
            results,
            key=lambda x: x["score"],
            reverse=True
        )

        return results


    return []


# ----------------------------------------------------------------
# 9. TEST SAMPLE
# ----------------------------------------------------------------
# Use a sample belonging to one of the disease categories
# present in the training dataset.
# ----------------------------------------------------------------

sample_summary = """
This randomized clinical trial evaluates a treatment
for adult patients with type 2 diabetes mellitus.
The study investigates changes in blood glucose,
glycemic control, HbA1c levels, treatment safety,
and diabetes-related clinical outcomes.
"""


print("\n" + "=" * 70)
print("SAMPLE CLINICAL TRIAL")
print("=" * 70)

print("\nOriginal Summary:")

print(
    sample_summary
)


# ----------------------------------------------------------------
# 10. MAKE PREDICTION
# ----------------------------------------------------------------

result = predict_disease_category(
    sample_summary
)


print("\nCleaned Summary:")

print(
    result["cleaned_text"]
)


print("\n" + "=" * 70)
print("PREDICTION RESULT")
print("=" * 70)


print("\nPredicted Disease Category:")

print(
    result["predicted_category"]
)


print(
    "\nModel Used:",
    model_metadata["selected_model"]
)


# ----------------------------------------------------------------
# 11. DISPLAY PREDICTION SCORES
# ----------------------------------------------------------------

print("\n" + "=" * 70)
print("MODEL PREDICTION INFORMATION")
print("=" * 70)


prediction_scores = get_prediction_scores(
    result["text_vector"]
)


if prediction_scores:

    score_type = prediction_scores[0][
        "score_type"
    ]

    if score_type == "probability":

        print(
            "\nCategory Probabilities:"
        )

    else:

        print(
            "\nCategory Decision Scores:"
        )


    for item in prediction_scores:

        print(
            f"{item['disease_category']}: "
            f"{item['score']:.4f}"
        )

else:

    print(
        "\nPrediction scores are not "
        "available for this model."
    )


# ----------------------------------------------------------------
# 12. TOP PREDICTION
# ----------------------------------------------------------------

if prediction_scores:

    top_result = prediction_scores[0]

    print("\nTop Prediction:")

    print(
        top_result[
            "disease_category"
        ]
    )

    if (
        top_result["score_type"]
        == "probability"
    ):

        print(
            "Probability:",
            f"{top_result['score']:.4f}"
        )

    else:

        print(
            "Decision Score:",
            f"{top_result['score']:.4f}"
        )


# ----------------------------------------------------------------
# 13. FINAL MESSAGE
# ----------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 7 COMPLETED SUCCESSFULLY")
print("=" * 70)


print(
    "\nThe trained NLP pipeline can classify "
    "new clinical trial summaries into the "
    "supported disease categories."
)
