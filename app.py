# ================================================================
# STREAMLIT APPLICATION
# Clinical Trial Disease Category Classification Using NLP
# ================================================================

import os
import joblib
import pandas as pd
import streamlit as st
import plotly.express as px

from text_preprocessing import clean_medical_text


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

FIG_DIR = "figures"


# ================================================================
# APPLICATION TITLE
# ================================================================

st.title("🧬 Clinical Trial Disease Category Classification")

st.markdown(
    """
    ### NLP and Machine Learning Based Clinical Trial Classification

    This application analyzes a clinical trial's **brief summary**
    and predicts its corresponding **disease category** using a
    trained Natural Language Processing and Machine Learning pipeline.
    """
)


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
        "Unable to load the trained model files."
    )

    st.exception(e)

    st.stop()


# ================================================================
# SIDEBAR
# ================================================================

st.sidebar.title("Navigation")

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

    st.header("📊 Project Overview")

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
            metadata["selected_model"]
        )

    st.divider()

    st.subheader("🎯 Project Objective")

    st.write(
        """
        The objective of this project is to automatically classify
        clinical trials into disease categories using Natural Language
        Processing and supervised Machine Learning.
        """
    )

    st.subheader("🔄 Machine Learning Pipeline")

    st.markdown(
        """
        **Clinical Trial Data**
        → **Text Preprocessing**
        → **EDA**
        → **TF-IDF Feature Extraction**
        → **Model Training**
        → **Model Evaluation**
        → **Disease Category Prediction**
        """
    )

    st.subheader("🧠 Machine Learning Models")

    models_df = pd.DataFrame({
        "Model": [
            "Logistic Regression",
            "Multinomial Naive Bayes",
            "Linear SVM"
        ],
        "Purpose": [
            "Linear multiclass text classification",
            "Probabilistic text classification",
            "Linear margin-based classification"
        ]
    })

    st.dataframe(
        models_df,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("📚 NLP Technique")

    st.write(
        """
        TF-IDF (Term Frequency–Inverse Document Frequency) is used
        to convert clinical trial summaries into numerical feature
        vectors that can be processed by machine-learning algorithms.
        """
    )

    st.warning(
        """
        ⚠️ This application is intended for educational and research
        purposes only. Predictions should not be used for medical
        diagnosis, treatment decisions, patient care, or clinical
        decision-making.
        """
    )


# ================================================================
# EDA PAGE
# ================================================================

elif page == "EDA":

    st.header("📈 Exploratory Data Analysis")

    # ------------------------------------------------------------
    # Category distribution
    # ------------------------------------------------------------

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
            "categoryorder": "total ascending"
        }
    )

    st.plotly_chart(
        fig_category,
        use_container_width=True
    )

    # ------------------------------------------------------------
    # Summary length
    # ------------------------------------------------------------

    st.subheader(
        "2. Clinical Trial Summary Length"
    )

    summary_length = (
        df["brief_summary"]
        .fillna("")
        .astype(str)
        .str.len()
    )

    length_df = pd.DataFrame({
        "summary_length": summary_length
    })

    fig_length = px.histogram(
        length_df,
        x="summary_length",
        nbins=50,
        title="Distribution of Brief Summary Length"
    )

    fig_length.update_layout(
        xaxis_title="Character Count",
        yaxis_title="Number of Trials"
    )

    st.plotly_chart(
        fig_length,
        use_container_width=True
    )

    # ------------------------------------------------------------
    # Top medical terms
    # ------------------------------------------------------------

    st.subheader(
        "3. Most Frequent Medical Terms"
    )

    if os.path.exists(TOP_TERMS_PATH):

        top_terms = pd.read_csv(
            TOP_TERMS_PATH
        )

        top_terms = top_terms.head(20)

        fig_terms = px.bar(
            top_terms.sort_values("frequency"),
            x="frequency",
            y="term",
            orientation="h",
            title="Top 20 Medical Terms"
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

        st.info(
            "top_terms_overall.csv was not found."
        )

    # ------------------------------------------------------------
    # Summary length by category
    # ------------------------------------------------------------

    st.subheader(
        "4. Summary Length by Disease Category"
    )

    category_length = (
        df.groupby("disease_category")["brief_summary"]
        .apply(
            lambda x: x.fillna("")
            .astype(str)
            .str.len()
            .mean()
        )
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
        title="Average Summary Length by Disease Category"
    )

    st.plotly_chart(
        fig_category_length,
        use_container_width=True
    )

    # ------------------------------------------------------------
    # Dataset preview
    # ------------------------------------------------------------

    st.subheader(
        "5. Dataset Preview"
    )

    display_columns = [
        "nct_id",
        "brief_summary",
        "disease_category"
    ]

    available_columns = [
        col
        for col in display_columns
        if col in df.columns
    ]

    st.dataframe(
        df[available_columns].head(20),
        use_container_width=True,
        hide_index=True
    )


# ================================================================
# DISEASE PREDICTION PAGE
# ================================================================

elif page == "Disease Prediction":

    st.header("🧠 Disease Category Prediction")

    st.write(
        """
        Enter a clinical trial's brief summary below.
        The trained NLP pipeline will preprocess the text,
        convert it into TF-IDF features, and predict the
        disease category.
        """
    )

    sample_text = """
    This randomized clinical trial evaluates the safety and efficacy
    of a new treatment in adult patients with heart failure and
    cardiovascular disease. The study measures cardiac function,
    hospitalization, mortality, and other cardiovascular outcomes.
    """

    text_input = st.text_area(
        "Clinical Trial Brief Summary",
        value="",
        height=250,
        placeholder="Paste the clinical trial brief summary here..."
    )

    col1, col2 = st.columns([1, 1])

    with col1:

        predict_button = st.button(
            "🔍 Predict Disease Category",
            type="primary",
            use_container_width=True
        )

    with col2:

        sample_button = st.button(
            "📝 Load Sample",
            use_container_width=True
        )

    if sample_button:

        text_input = sample_text

        st.info(
            "Sample loaded. Click 'Predict Disease Category'."
        )

    if predict_button:

        if not text_input.strip():

            st.warning(
                "Please enter a clinical trial summary."
            )

        else:

            with st.spinner(
                "Analyzing clinical trial summary..."
            ):

                # ------------------------------------------------
                # Text preprocessing
                # ------------------------------------------------

                cleaned_text = clean_medical_text(
                    text_input
                )

                if not cleaned_text.strip():

                    st.error(
                        "No usable text remains after preprocessing."
                    )

                else:

                    # --------------------------------------------
                    # TF-IDF transformation
                    # --------------------------------------------

                    text_vector = vectorizer.transform(
                        [cleaned_text]
                    )

                    # --------------------------------------------
                    # Prediction
                    # --------------------------------------------

                    prediction_encoded = model.predict(
                        text_vector
                    )[0]

                    prediction = (
                        label_encoder
                        .inverse_transform(
                            [prediction_encoded]
                        )[0]
                    )

                    # --------------------------------------------
                    # Display prediction
                    # --------------------------------------------

                    st.success(
                        f"Predicted Disease Category: "
                        f"{prediction}"
                    )

                    st.metric(
                        "Predicted Category",
                        prediction
                    )

                    # --------------------------------------------
                    # Model information
                    # --------------------------------------------

                    st.info(
                        f"Model used: "
                        f"{metadata['selected_model']}"
                    )

                    # --------------------------------------------
                    # Cleaned text
                    # --------------------------------------------

                    with st.expander(
                        "🔎 View Preprocessed Text"
                    ):

                        st.write(
                            cleaned_text
                        )

                    # --------------------------------------------
                    # Probability information
                    # --------------------------------------------

                    if hasattr(
                        model,
                        "predict_proba"
                    ):

                        probabilities = (
                            model
                            .predict_proba(
                                text_vector
                            )[0]
                        )

                        probability_df = pd.DataFrame({
                            "Disease Category":
                                label_encoder.classes_,
                            "Probability":
                                probabilities
                        })

                        probability_df = (
                            probability_df
                            .sort_values(
                                "Probability",
                                ascending=False
                            )
                            .head(10)
                        )

                        st.subheader(
                            "Prediction Probability"
                        )

                        fig_probability = px.bar(
                            probability_df.sort_values(
                                "Probability"
                            ),
                            x="Probability",
                            y="Disease Category",
                            orientation="h",
                            text="Probability"
                        )

                        fig_probability.update_traces(
                            texttemplate="%{text:.2%}"
                        )

                        fig_probability.update_layout(
                            xaxis_title="Probability",
                            yaxis_title="Disease Category",
                            xaxis_tickformat=".0%"
                        )

                        st.plotly_chart(
                            fig_probability,
                            use_container_width=True
                        )

                    else:

                        st.info(
                            """
                            The selected model does not provide
                            probability estimates. The prediction
                            is based on the model's decision function.
                            """
                        )

    st.divider()

    st.warning(
        """
        ⚠️ Prediction is for research and educational purposes only.
        It must not be interpreted as a medical diagnosis or used
        for clinical decision-making.
        """
    )


# ================================================================
# MODEL PERFORMANCE PAGE
# ================================================================

elif page == "Model Performance":

    st.header("📊 Model Performance")

    if os.path.exists(
        MODEL_COMPARISON_PATH
    ):

        results_df = pd.read_csv(
            MODEL_COMPARISON_PATH
        )

        st.subheader(
            "Model Comparison"
        )

        st.dataframe(
            results_df,
            use_container_width=True,
            hide_index=True
        )

        # --------------------------------------------------------
        # Performance chart
        # --------------------------------------------------------

        st.subheader(
            "Performance Comparison"
        )

        metric_columns = [
            "accuracy",
            "precision_weighted",
            "recall_weighted",
            "f1_weighted"
        ]

        available_metrics = [
            col
            for col in metric_columns
            if col in results_df.columns
        ]

        chart_df = results_df.melt(
            id_vars=["model"],
            value_vars=available_metrics,
            var_name="Metric",
            value_name="Score"
        )

        fig_performance = px.bar(
            chart_df,
            x="model",
            y="Score",
            color="Metric",
            barmode="group",
            title="Machine Learning Model Performance"
        )

        fig_performance.update_layout(
            yaxis_range=[0, 1]
        )

        st.plotly_chart(
            fig_performance,
            use_container_width=True
        )

        # --------------------------------------------------------
        # Selected model
        # --------------------------------------------------------

        st.subheader(
            "Selected Model"
        )

        st.success(
            f"Selected Model: "
            f"{metadata['selected_model']}"
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric(
                "Accuracy",
                f"{metadata['accuracy']:.2%}"
            )

        with col2:

            st.metric(
                "Weighted Precision",
                f"{metadata['precision_weighted']:.2%}"
            )

        with col3:

            st.metric(
                "Weighted Recall",
                f"{metadata['recall_weighted']:.2%}"
            )

        with col4:

            st.metric(
                "Weighted F1",
                f"{metadata['f1_weighted']:.2%}"
            )

    else:

        st.error(
            "model_comparison.csv was not found. "
            "Please run Step 6 first."
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