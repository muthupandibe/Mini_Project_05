# ================================================================
# STREAMLIT APPLICATION
# Clinical Trial Disease Category Classification Using NLP
# ================================================================

import os
import re
import joblib
import nltk
import pandas as pd
import streamlit as st
import plotly.express as px

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# ================================================================
# PAGE CONFIGURATION
# ================================================================

st.set_page_config(
    page_title="Clinical Trial Disease Classification",
    page_icon="🧬",
    layout="wide"
)


# ================================================================
# FILE PATHS
# ================================================================

DATA_PATH = "clinical_trials_cleaned.csv"

VECTORIZER_PATH = "tfidf_vectorizer.pkl"
MODEL_PATH = "selected_model.pkl"
ENCODER_PATH = "label_encoder.pkl"
METADATA_PATH = "model_metadata.pkl"

MODEL_COMPARISON_PATH = "model_comparison.csv"
TOP_TERMS_PATH = "top_terms_overall.csv"


# ================================================================
# NLTK RESOURCES
# ================================================================

@st.cache_resource
def load_nltk_resources():

    nltk.download(
        "stopwords",
        quiet=True
    )

    nltk.download(
        "wordnet",
        quiet=True
    )

    nltk.download(
        "omw-1.4",
        quiet=True
    )

    stop_words = set(
        stopwords.words("english")
    )

    # Preserve important negation words
    stop_words = stop_words - {
        "no",
        "not",
        "nor"
    }

    lemmatizer = WordNetLemmatizer()

    return stop_words, lemmatizer


stop_words, lemmatizer = load_nltk_resources()


# ================================================================
# TEXT PREPROCESSING FUNCTION
# ================================================================

def clean_medical_text(text):

    """
    Apply the same preprocessing used during model training.
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

    # Remove stopwords
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


# ================================================================
# LOAD MODEL COMPONENTS
# ================================================================

@st.cache_resource
def load_model_components():

    vectorizer = joblib.load(
        VECTORIZER_PATH
    )

    model = joblib.load(
        MODEL_PATH
    )

    label_encoder = joblib.load(
        ENCODER_PATH
    )

    metadata = joblib.load(
        METADATA_PATH
    )

    return (
        vectorizer,
        model,
        label_encoder,
        metadata
    )


# ================================================================
# LOAD DATASET
# ================================================================

@st.cache_data
def load_dataset():

    return pd.read_csv(
        DATA_PATH
    )


# ================================================================
# LOAD FILES SAFELY
# ================================================================

try:

    (
        vectorizer,
        model,
        label_encoder,
        metadata
    ) = load_model_components()

    df = load_dataset()

except Exception as e:

    st.error(
        "Unable to load the trained model or dataset files."
    )

    st.exception(e)

    st.stop()


# ================================================================
# APPLICATION TITLE
# ================================================================

st.title(
    "🧬 Clinical Trial Disease Category Classification"
)

st.markdown(
    """
    ### NLP and Machine Learning Based Clinical Trial Classification

    This application analyzes a clinical trial's **brief summary**
    and predicts its corresponding **disease category** using a
    trained Natural Language Processing and Machine Learning pipeline.
    """
)


# ================================================================
# SIDEBAR
# ================================================================

st.sidebar.title(
    "Navigation"
)

page = st.sidebar.radio(
    "Go to",
    [
        "Overview",
        "EDA",
        "Disease Prediction",
        "Model Performance"
    ]
)


# ================================================================
# OVERVIEW PAGE
# ================================================================

if page == "Overview":

    st.header(
        "📊 Project Overview"
    )


    # ------------------------------------------------------------
    # KPI CARDS
    # ------------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "Clinical Trials",
            f"{len(df):,}"
        )


    with col2:

        st.metric(
            "Disease Categories",
            df["disease_category"].nunique()
        )


    with col3:

        st.metric(
            "TF-IDF Features",
            f"{len(vectorizer.get_feature_names_out()):,}"
        )


    with col4:

        st.metric(
            "Selected Model",
            metadata.get(
                "selected_model",
                type(model).__name__
            )
        )


    st.divider()


    # ------------------------------------------------------------
    # PROJECT OBJECTIVE
    # ------------------------------------------------------------

    st.subheader(
        "🎯 Project Objective"
    )

    st.write(
        """
        The objective of this project is to automatically classify
        clinical trial brief summaries into appropriate disease
        categories using Natural Language Processing and supervised
        Machine Learning.
        """
    )


    # ------------------------------------------------------------
    # PIPELINE
    # ------------------------------------------------------------

    st.subheader(
        "🔄 Machine Learning Pipeline"
    )

    st.markdown(
        """
        **Clinical Trial Data**

        ↓

        **Text Preprocessing**

        ↓

        **Exploratory Data Analysis**

        ↓

        **TF-IDF Feature Extraction**

        ↓

        **Train-Test Split**

        ↓

        **Machine Learning Model Training**

        ↓

        **Model Evaluation**

        ↓

        **Disease Category Prediction**
        """
    )


    # ------------------------------------------------------------
    # MODELS
    # ------------------------------------------------------------

    st.subheader(
        "🧠 Machine Learning Models"
    )


    models_df = pd.DataFrame(
        {
            "Model": [
                "Logistic Regression",
                "Multinomial Naive Bayes",
                "Linear SVM"
            ],

            "Purpose": [
                "Linear multiclass text classification",
                "Probabilistic text classification",
                "Linear margin-based text classification"
            ]
        }
    )


    st.dataframe(
        models_df,
        use_container_width=True,
        hide_index=True
    )


    # ------------------------------------------------------------
    # NLP TECHNIQUE
    # ------------------------------------------------------------

    st.subheader(
        "📚 NLP Technique"
    )

    st.write(
        """
        TF-IDF (Term Frequency-Inverse Document Frequency)
        converts cleaned clinical trial summaries into numerical
        feature vectors that can be processed by Machine Learning
        classification algorithms.
        """
    )


    # ------------------------------------------------------------
    # SUPPORTED CATEGORIES
    # ------------------------------------------------------------

    st.subheader(
        "🏷️ Supported Disease Categories"
    )


    supported_categories = pd.DataFrame(
        {
            "Disease Category":
                label_encoder.classes_
        }
    )


    st.dataframe(
        supported_categories,
        use_container_width=True,
        hide_index=True
    )


    st.info(
        """
        The classifier can only predict disease categories
        that were included in the training dataset.
        """
    )


    # ------------------------------------------------------------
    # DISCLAIMER
    # ------------------------------------------------------------

    st.warning(
        """
        ⚠️ This application is intended for educational and research
        purposes only.

        Predictions should not be used for medical diagnosis,
        treatment decisions, patient care, or clinical
        decision-making.
        """
    )


# ================================================================
# EDA PAGE
# ================================================================

elif page == "EDA":

    st.header(
        "📈 Exploratory Data Analysis"
    )


    # ============================================================
    # 1. DISEASE CATEGORY DISTRIBUTION
    # ============================================================

    st.subheader(
        "1. Clinical Trial Distribution by Disease Category"
    )


    category_counts = (
        df["disease_category"]
        .value_counts()
        .reset_index()
    )


    category_counts.columns = [
        "disease_category",
        "trial_count"
    ]


    fig_category = px.bar(
        category_counts,
        x="trial_count",
        y="disease_category",
        orientation="h",
        title="Clinical Trial Count by Disease Category",
        text="trial_count"
    )


    fig_category.update_layout(
        yaxis={
            "categoryorder":
                "total ascending"
        },

        xaxis_title="Number of Clinical Trials",

        yaxis_title="Disease Category"
    )


    st.plotly_chart(
        fig_category,
        use_container_width=True
    )


    st.dataframe(
        category_counts,
        use_container_width=True,
        hide_index=True
    )


    # ============================================================
    # 2. SUMMARY LENGTH DISTRIBUTION
    # ============================================================

    st.subheader(
        "2. Clinical Trial Summary Length"
    )


    summary_length = (
        df["brief_summary"]
        .fillna("")
        .astype(str)
        .str.len()
    )


    length_df = pd.DataFrame(
        {
            "summary_length":
                summary_length
        }
    )


    fig_length = px.histogram(
        length_df,
        x="summary_length",
        nbins=50,
        title="Distribution of Brief Summary Length"
    )


    fig_length.update_layout(
        xaxis_title="Character Count",
        yaxis_title="Number of Clinical Trials"
    )


    st.plotly_chart(
        fig_length,
        use_container_width=True
    )


    # ============================================================
    # 3. TOP MEDICAL TERMS
    # ============================================================

    st.subheader(
        "3. Most Frequent Medical Terms"
    )


    if os.path.exists(
        TOP_TERMS_PATH
    ):

        top_terms = pd.read_csv(
            TOP_TERMS_PATH
        )


        top_terms = top_terms.head(
            20
        )


        # --------------------------------------------------------
        # FREQUENCY-BASED FILE
        # --------------------------------------------------------

        if (
            "term" in top_terms.columns
            and
            "frequency" in top_terms.columns
        ):

            plot_terms = (
                top_terms
                .sort_values(
                    "frequency"
                )
            )


            fig_terms = px.bar(
                plot_terms,
                x="frequency",
                y="term",
                orientation="h",
                title="Top 20 Frequently Occurring Medical Terms"
            )


            fig_terms.update_layout(
                xaxis_title="Frequency",
                yaxis_title="Medical Term"
            )


            st.plotly_chart(
                fig_terms,
                use_container_width=True
            )


            st.dataframe(
                top_terms,
                use_container_width=True,
                hide_index=True
            )


        # --------------------------------------------------------
        # TF-IDF-BASED FILE
        # --------------------------------------------------------

        elif (
            "term" in top_terms.columns
            and
            "mean_tfidf" in top_terms.columns
        ):

            plot_terms = (
                top_terms
                .sort_values(
                    "mean_tfidf"
                )
            )


            fig_terms = px.bar(
                plot_terms,
                x="mean_tfidf",
                y="term",
                orientation="h",
                title="Top 20 Terms by Mean TF-IDF Score"
            )


            fig_terms.update_layout(
                xaxis_title="Mean TF-IDF Score",
                yaxis_title="Medical Term"
            )


            st.plotly_chart(
                fig_terms,
                use_container_width=True
            )


            st.dataframe(
                top_terms,
                use_container_width=True,
                hide_index=True
            )


        else:

            st.warning(
                "top_terms_overall.csv does not contain "
                "the expected columns."
            )


    else:

        st.info(
            "top_terms_overall.csv was not found. "
            "Run the EDA step first."
        )


    # ============================================================
    # 4. SUMMARY LENGTH BY DISEASE CATEGORY
    # ============================================================

    st.subheader(
        "4. Summary Length by Disease Category"
    )


    category_length_df = df.copy()


    category_length_df[
        "summary_length"
    ] = (
        category_length_df[
            "brief_summary"
        ]
        .fillna("")
        .astype(str)
        .str.len()
    )


    category_length = (
        category_length_df
        .groupby(
            "disease_category"
        )["summary_length"]
        .mean()
        .reset_index()
    )


    category_length.columns = [
        "disease_category",
        "average_summary_length"
    ]


    fig_category_length = px.bar(
        category_length.sort_values(
            "average_summary_length"
        ),
        x="average_summary_length",
        y="disease_category",
        orientation="h",
        title=(
            "Average Summary Length "
            "by Disease Category"
        )
    )


    fig_category_length.update_layout(
        xaxis_title=(
            "Average Summary Length "
            "(Characters)"
        ),
        yaxis_title="Disease Category"
    )


    st.plotly_chart(
        fig_category_length,
        use_container_width=True
    )


    st.dataframe(
        category_length,
        use_container_width=True,
        hide_index=True
    )


    # ============================================================
    # 5. DATASET PREVIEW
    # ============================================================

    st.subheader(
        "5. Dataset Preview"
    )


    display_columns = [
        "nct_id",
        "brief_summary",
        "disease_category"
    ]


    available_columns = [
        column
        for column in display_columns
        if column in df.columns
    ]


    st.dataframe(
        df[
            available_columns
        ].head(20),
        use_container_width=True,
        hide_index=True
    )


# ================================================================
# DISEASE PREDICTION PAGE
# ================================================================

elif page == "Disease Prediction":

    st.header(
        "🧠 Disease Category Prediction"
    )


    st.write(
        """
        Enter or paste a clinical trial's brief summary below.

        The trained NLP pipeline will preprocess the text,
        transform it into TF-IDF features, and predict the
        corresponding disease category.
        """
    )


    st.info(
        """
        The model can only predict disease categories
        that were included in its training dataset.
        """
    )


    # ============================================================
    # USER INPUT
    # ============================================================

    text_input = st.text_area(
        "Clinical Trial Brief Summary",
        height=250,
        placeholder=(
            "Enter or paste the clinical trial "
            "brief summary here..."
        )
    )


    # ============================================================
    # PREDICT BUTTON
    # ============================================================

    predict_button = st.button(
        "🔍 Predict Disease Category",
        type="primary",
        use_container_width=True
    )


    # ============================================================
    # PREDICTION
    # ============================================================

    if predict_button:


        # --------------------------------------------------------
        # EMPTY INPUT CHECK
        # --------------------------------------------------------

        if not text_input.strip():

            st.warning(
                "Please enter a clinical trial summary."
            )


        else:

            with st.spinner(
                "Analyzing clinical trial summary..."
            ):

                try:

                    # ============================================
                    # STEP 1: TEXT PREPROCESSING
                    # ============================================

                    cleaned_text = clean_medical_text(
                        text_input
                    )


                    if not cleaned_text.strip():

                        st.error(
                            "No usable text remains "
                            "after preprocessing."
                        )


                    else:

                        # ========================================
                        # STEP 2: TF-IDF TRANSFORMATION
                        # ========================================

                        text_vector = (
                            vectorizer.transform(
                                [cleaned_text]
                            )
                        )


                        # ========================================
                        # STEP 3: MODEL PREDICTION
                        # ========================================

                        prediction_encoded = (
                            model.predict(
                                text_vector
                            )[0]
                        )


                        # ========================================
                        # STEP 4: DECODE PREDICTION
                        # ========================================

                        prediction = (
                            label_encoder
                            .inverse_transform(
                                [
                                    prediction_encoded
                                ]
                            )[0]
                        )


                        # ========================================
                        # STEP 5: DISPLAY PREDICTION
                        # ========================================

                        st.success(
                            f"Predicted Disease Category: "
                            f"{prediction}"
                        )


                        col1, col2 = st.columns(2)


                        with col1:

                            st.metric(
                                "Predicted Disease Category",
                                prediction
                            )


                        with col2:

                            st.metric(
                                "Model Used",
                                metadata.get(
                                    "selected_model",
                                    type(model).__name__
                                )
                            )


                        # ========================================
                        # STEP 6: PREPROCESSED TEXT
                        # ========================================

                        with st.expander(
                            "🔎 View Preprocessed Text"
                        ):

                            st.write(
                                cleaned_text
                            )


                        # ========================================
                        # STEP 7A: PROBABILITY-BASED MODELS
                        # ========================================

                        if hasattr(
                            model,
                            "predict_proba"
                        ):

                            probabilities = (
                                model.predict_proba(
                                    text_vector
                                )[0]
                            )


                            probability_df = pd.DataFrame(
                                {
                                    "Disease Category":
                                        label_encoder.classes_,

                                    "Probability":
                                        probabilities
                                }
                            )


                            probability_df = (
                                probability_df
                                .sort_values(
                                    "Probability",
                                    ascending=False
                                )
                                .reset_index(
                                    drop=True
                                )
                            )


                            top_probability = (
                                probability_df
                                .iloc[0][
                                    "Probability"
                                ]
                            )


                            st.subheader(
                                "📊 Prediction Confidence"
                            )


                            st.metric(
                                "Top Prediction Probability",
                                f"{top_probability:.2%}"
                            )


                            # ------------------------------------
                            # PROBABILITY CHART
                            # ------------------------------------

                            fig_probability = px.bar(

                                probability_df.sort_values(
                                    "Probability"
                                ),

                                x="Probability",

                                y="Disease Category",

                                orientation="h",

                                text="Probability",

                                title=(
                                    "Disease Category "
                                    "Prediction Probabilities"
                                )
                            )


                            fig_probability.update_traces(
                                texttemplate="%{text:.2%}"
                            )


                            fig_probability.update_layout(

                                xaxis_title="Probability",

                                yaxis_title=(
                                    "Disease Category"
                                ),

                                xaxis_tickformat=".0%"
                            )


                            st.plotly_chart(
                                fig_probability,
                                use_container_width=True
                            )


                            # ------------------------------------
                            # PROBABILITY TABLE
                            # ------------------------------------

                            display_probability_df = (
                                probability_df.copy()
                            )


                            display_probability_df[
                                "Probability"
                            ] = (

                                display_probability_df[
                                    "Probability"
                                ]

                                .map(
                                    lambda value:
                                    f"{value:.2%}"
                                )
                            )


                            st.dataframe(
                                display_probability_df,
                                use_container_width=True,
                                hide_index=True
                            )


                        # ========================================
                        # STEP 7B: LINEAR SVM
                        # ========================================

                        elif hasattr(
                            model,
                            "decision_function"
                        ):

                            decision_scores = (
                                model.decision_function(
                                    text_vector
                                )[0]
                            )


                            decision_df = pd.DataFrame(
                                {
                                    "Disease Category":
                                        label_encoder.classes_,

                                    "Decision Score":
                                        decision_scores
                                }
                            )


                            decision_df = (
                                decision_df
                                .sort_values(
                                    "Decision Score",
                                    ascending=False
                                )
                                .reset_index(
                                    drop=True
                                )
                            )


                            st.subheader(
                                "📊 Model Decision Scores"
                            )


                            st.info(
                                """
                                Linear SVM does not provide direct
                                probability estimates.

                                The values shown below are decision
                                scores. A higher score indicates a
                                stronger model preference for that
                                disease category.

                                Decision scores are not probabilities.
                                """
                            )


                            top_decision_score = (
                                decision_df
                                .iloc[0][
                                    "Decision Score"
                                ]
                            )


                            st.metric(
                                "Top Decision Score",
                                f"{top_decision_score:.4f}"
                            )


                            # ------------------------------------
                            # DECISION SCORE CHART
                            # ------------------------------------

                            fig_decision = px.bar(

                                decision_df.sort_values(
                                    "Decision Score"
                                ),

                                x="Decision Score",

                                y="Disease Category",

                                orientation="h",

                                title=(
                                    "Disease Category "
                                    "Decision Scores"
                                )
                            )


                            fig_decision.update_layout(

                                xaxis_title=(
                                    "Decision Score"
                                ),

                                yaxis_title=(
                                    "Disease Category"
                                )
                            )


                            st.plotly_chart(
                                fig_decision,
                                use_container_width=True
                            )


                            # ------------------------------------
                            # DECISION SCORE TABLE
                            # ------------------------------------

                            st.dataframe(
                                decision_df,
                                use_container_width=True,
                                hide_index=True
                            )


                        # ========================================
                        # STEP 7C: OTHER MODEL TYPE
                        # ========================================

                        else:

                            st.info(
                                "Prediction confidence "
                                "information is not available "
                                "for the selected model."
                            )


                except Exception as e:

                    st.error(
                        "An error occurred while "
                        "making the prediction."
                    )

                    st.exception(e)


    # ============================================================
    # DISCLAIMER
    # ============================================================

    st.divider()


    st.warning(
        """
        ⚠️ This prediction is intended for educational and
        research purposes only.

        The model is restricted to disease categories represented
        in its training dataset.

        Results must not be interpreted as medical diagnoses or
        used for clinical decision-making.
        """
    )


# ================================================================
# MODEL PERFORMANCE PAGE
# ================================================================

elif page == "Model Performance":

    st.header(
        "📊 Model Performance"
    )


    # ============================================================
    # LOAD MODEL COMPARISON
    # ============================================================

    if os.path.exists(
        MODEL_COMPARISON_PATH
    ):

        results_df = pd.read_csv(
            MODEL_COMPARISON_PATH
        )


        # --------------------------------------------------------
        # SORT BY MACRO F1
        # --------------------------------------------------------

        if "f1_macro" in results_df.columns:

            results_df = (
                results_df
                .sort_values(
                    by="f1_macro",
                    ascending=False
                )
                .reset_index(
                    drop=True
                )
            )


        # ========================================================
        # MODEL COMPARISON TABLE
        # ========================================================

        st.subheader(
            "1. Model Comparison"
        )


        st.dataframe(
            results_df,
            use_container_width=True,
            hide_index=True
        )


        # ========================================================
        # PERFORMANCE CHART
        # ========================================================

        st.subheader(
            "2. Performance Comparison"
        )


        metric_columns = [
            "accuracy",

            "precision_weighted",
            "recall_weighted",
            "f1_weighted",

            "precision_macro",
            "recall_macro",
            "f1_macro"
        ]


        available_metrics = [
            column
            for column in metric_columns
            if column in results_df.columns
        ]


        if available_metrics:

            chart_df = results_df.melt(

                id_vars=[
                    "model"
                ],

                value_vars=
                    available_metrics,

                var_name="Metric",

                value_name="Score"
            )


            fig_performance = px.bar(

                chart_df,

                x="model",

                y="Score",

                color="Metric",

                barmode="group",

                title=(
                    "Machine Learning Model "
                    "Performance Comparison"
                )
            )


            fig_performance.update_layout(

                yaxis_range=[
                    0,
                    1
                ],

                xaxis_title=(
                    "Machine Learning Model"
                ),

                yaxis_title=(
                    "Performance Score"
                )
            )


            st.plotly_chart(
                fig_performance,
                use_container_width=True
            )


        # ========================================================
        # SELECTED MODEL
        # ========================================================

        st.subheader(
            "3. Selected Model"
        )


        selected_model_name = metadata.get(
            "selected_model",
            type(model).__name__
        )


        st.success(
            f"Selected Model: "
            f"{selected_model_name}"
        )


        selection_metric = metadata.get(
            "selection_metric",
            "Macro F1 Score"
        )


        st.write(
            f"**Model Selection Metric:** "
            f"{selection_metric}"
        )


        # ========================================================
        # MAIN METRICS
        # ========================================================

        st.subheader(
            "4. Selected Model Evaluation"
        )


        col1, col2, col3 = st.columns(
            3
        )


        with col1:

            if "accuracy" in metadata:

                st.metric(
                    "Accuracy",
                    f"{metadata['accuracy']:.2%}"
                )


        with col2:

            if "f1_weighted" in metadata:

                st.metric(
                    "Weighted F1",
                    f"{metadata['f1_weighted']:.2%}"
                )


        with col3:

            if "f1_macro" in metadata:

                st.metric(
                    "Macro F1",
                    f"{metadata['f1_macro']:.2%}"
                )


        # ========================================================
        # WEIGHTED METRICS
        # ========================================================

        weighted_keys = [
            "precision_weighted",
            "recall_weighted",
            "f1_weighted"
        ]


        if all(
            key in metadata
            for key in weighted_keys
        ):

            st.subheader(
                "5. Weighted Evaluation Metrics"
            )


            col1, col2, col3 = st.columns(
                3
            )


            with col1:

                st.metric(
                    "Weighted Precision",
                    f"{metadata['precision_weighted']:.2%}"
                )


            with col2:

                st.metric(
                    "Weighted Recall",
                    f"{metadata['recall_weighted']:.2%}"
                )


            with col3:

                st.metric(
                    "Weighted F1",
                    f"{metadata['f1_weighted']:.2%}"
                )


        # ========================================================
        # MACRO METRICS
        # ========================================================

        macro_keys = [
            "precision_macro",
            "recall_macro",
            "f1_macro"
        ]


        if all(
            key in metadata
            for key in macro_keys
        ):

            st.subheader(
                "6. Macro Evaluation Metrics"
            )


            col1, col2, col3 = st.columns(
                3
            )


            with col1:

                st.metric(
                    "Macro Precision",
                    f"{metadata['precision_macro']:.2%}"
                )


            with col2:

                st.metric(
                    "Macro Recall",
                    f"{metadata['recall_macro']:.2%}"
                )


            with col3:

                st.metric(
                    "Macro F1",
                    f"{metadata['f1_macro']:.2%}"
                )


            st.info(
                """
                Macro metrics give equal importance to every
                disease category.

                They are useful for this project because the
                clinical trial dataset contains imbalanced
                disease categories.
                """
            )


    else:

        st.error(
            "model_comparison.csv was not found. "
            "Please run Step 6 Model Evaluation first."
        )


# ================================================================
# FOOTER
# ================================================================

st.divider()


st.caption(
    """
    Clinical Trial Disease Category Classification |
    NLP + TF-IDF + Machine Learning |
    Educational / Research Use Only
    """
)
