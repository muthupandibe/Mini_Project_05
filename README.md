# Clinical Trial Disease Category Classification Using NLP and Machine Learning

## 1. Project Overview

The **Clinical Trial Disease Category Classification Using NLP and Machine Learning** project is a healthcare analytics application that uses **Natural Language Processing (NLP)** and **Machine Learning** to automatically classify clinical trials into disease categories based on their clinical trial descriptions.

The system analyzes the `brief_summary` of each clinical trial, preprocesses the medical text, converts the text into numerical **TF-IDF features**, trains multiple machine learning classification models, evaluates their performance, and uses the selected model to predict the disease category of new clinical trial summaries.

The project also includes a **Streamlit web application** that provides an interactive interface for disease-category prediction and model-performance visualization.

---

## 2. Project Objectives

The main objectives of this project are:

* Collect and inspect clinical trial data.
* Extract relevant clinical trial information.
* Clean and preprocess medical text using NLP techniques.
* Explore disease-category distributions and clinical-trial text patterns.
* Identify frequently occurring medical terms.
* Convert clinical text into numerical features using TF-IDF.
* Train multiple machine learning classification models.
* Compare model performance using standard evaluation metrics.
* Automatically select the model with the highest weighted F1-score.
* Predict disease categories for new clinical trial descriptions.
* Generate insights and reports for healthcare analytics and medical research.
* Provide an interactive Streamlit application for prediction and analysis.

---

## 3. Dataset

### Input Dataset

The project uses the following raw dataset:

```text
clinical_trials_raw_patient2trial_conditions_new.csv
```

### Important Columns

| Column                   | Description                             |
| ------------------------ | --------------------------------------- |
| `nct_id`                 | Unique clinical trial identifier        |
| `brief_summary`          | Brief description of the clinical trial |
| `source_condition_query` | Original disease/condition category     |

During Step 1, `source_condition_query` is renamed to:

```text
disease_category
```

The main NLP input is:

```text
brief_summary
```

The prediction target is:

```text
disease_category
```

---

## 4. Technologies Used

### Programming Language

* Python 3.13

### Libraries

* Pandas
* NumPy
* Scikit-learn
* NLTK
* Matplotlib
* Seaborn
* Joblib
* Streamlit
* Plotly

### Machine Learning Techniques

* TF-IDF Vectorization
* Logistic Regression
* Multinomial Naive Bayes
* Linear Support Vector Machine (LinearSVC)

---

## 5. Project Workflow

```text
Raw Clinical Trial Dataset
          |
          v
Step 1 - Data Collection
          |
          v
Step 2 - Text Preprocessing
          |
          v
Step 3 - Exploratory Data Analysis
          |
          v
Step 4 - TF-IDF Feature Extraction
          |
          v
Step 5 - Model Training
          |
          v
Step 6 - Model Evaluation
          |
          v
Step 7 - Insights & Reporting
          |
          v
Streamlit Application
          |
          v
Disease Category Prediction
```

---

# 6. Step 1 - Data Collection

File:

```text
step1_data_collection.py
```

### Purpose

Step 1 loads the raw clinical trial dataset and performs an initial inspection.

The script checks:

* Dataset shape
* Column names
* Data types
* First few records
* Missing values
* Duplicate trial IDs
* Sample clinical-trial summary
* Average summary length
* Number of disease categories
* Disease-category distribution

### Selected Columns

The following columns are used for the NLP pipeline:

```python
df_selected = df[
    ["nct_id", "brief_summary", "source_condition_query"]
].rename(
    columns={
        "source_condition_query": "disease_category"
    }
)
```

### Output

```text
clinical_trials_selected.csv
```

---

# 7. Step 2 - Text Preprocessing

File:

```text
step2_preprocessing.py
```

### Purpose

Clinical trial descriptions contain natural-language text that must be cleaned before machine learning.

The preprocessing pipeline performs:

1. Missing-value handling
2. Duplicate trial removal
3. Lowercase conversion
4. HTML removal
5. URL removal
6. Special-character removal
7. Extra-space removal
8. Tokenization
9. Stopword removal
10. Negation-word preservation
11. Word lemmatization

### Important Preserved Words

The following negation words are retained:

```text
no
not
nor
```

This is useful because negation can be important when interpreting medical text.

### Example

Original:

```text
Patients with cardiovascular disease are NOT eligible for this study.
```

After preprocessing, the text is converted into a simplified normalized representation while preserving important terms such as `not`.

### Output

```text
clinical_trials_cleaned.csv
```

Important generated columns include:

```text
nct_id
brief_summary
disease_category
cleaned_summary
original_text_length
cleaned_text_length
word_count
```

---

# 8. Shared Text Preprocessing Module

File:

```text
text_preprocessing.py
```

This module contains the reusable function:

```python
clean_medical_text()
```

The same preprocessing logic is used during prediction so that the text entered into the Streamlit application is processed consistently with the text used during model training.

This consistency is important because the trained TF-IDF vectorizer expects text prepared using the same preprocessing pipeline.

---

# 9. Step 3 - Exploratory Data Analysis

File:

```text
step3_eda.py
```

### Purpose

Step 3 explores the clinical trial dataset and identifies important patterns.

### Analysis Performed

#### Disease Category Distribution

The number of clinical trials in each disease category is calculated.

Output:

```text
disease_category_distribution.csv
```

#### Summary Length Distribution

The character length of clinical-trial summaries is analyzed.

#### Frequently Occurring Terms

The most frequently occurring terms in the cleaned clinical-trial summaries are identified.

Output:

```text
top_terms_overall.csv
```

#### Category-Level Terms

Frequently occurring terms are also analyzed separately for different disease categories.

#### Summary Length by Disease Category

The project compares clinical-trial summary lengths across disease categories.

Output:

```text
average_summary_length_by_category.csv
```

#### EDA Summary

General EDA statistics are saved in:

```text
eda_summary.csv
```

### Generated Figures

```text
figures/
├── category_distribution.png
├── summary_length_distribution.png
├── top_terms_overall.png
├── top_terms_by_category.png
├── summary_length_by_category.png
└── average_summary_length_by_category.png
```

---

# 10. Step 4 - TF-IDF Feature Extraction

File:

```text
step4_tfidf.py
```

### Purpose

Machine learning algorithms require numerical input. Since clinical trial summaries are text, the project converts the cleaned text into numerical features using **TF-IDF (Term Frequency-Inverse Document Frequency)**.

### TF-IDF Configuration

```python
TfidfVectorizer(
    max_features=20000,
    min_df=2,
    max_df=0.95,
    ngram_range=(1, 2),
    sublinear_tf=True,
    strip_accents="unicode"
)
```

### Parameters

| Parameter                 | Purpose                                          |
| ------------------------- | ------------------------------------------------ |
| `max_features=20000`      | Limits the vocabulary size                       |
| `min_df=2`                | Removes extremely rare terms                     |
| `max_df=0.95`             | Removes terms occurring in almost every document |
| `ngram_range=(1,2)`       | Uses unigrams and bigrams                        |
| `sublinear_tf=True`       | Applies logarithmic term-frequency scaling       |
| `strip_accents="unicode"` | Normalizes accented characters                   |

### Output

The trained vectorizer is saved as:

```text
tfidf_vectorizer.pkl
```

The top TF-IDF features are saved as:

```text
top_tfidf_features.csv
```

The TF-IDF matrix remains sparse to reduce unnecessary memory usage.

---

# 11. Step 5 - Model Training

File:

```text
step5_train_models.py
```

Three machine learning models are trained.

## 11.1 Logistic Regression

```python
LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    random_state=42
)
```

Logistic Regression is a widely used linear classification algorithm that works effectively with high-dimensional sparse text features.

---

## 11.2 Multinomial Naive Bayes

```python
MultinomialNB(alpha=0.5)
```

Multinomial Naive Bayes is commonly used for text classification because it works naturally with word-frequency and TF-IDF-style features.

---

## 11.3 Linear SVM

```python
LinearSVC(
    class_weight="balanced",
    random_state=42
)
```

Linear SVM is suitable for high-dimensional text classification problems and often performs well with sparse TF-IDF features.

---

## Train-Test Split

The dataset is divided into:

```text
80% Training
20% Testing
```

The split uses stratification to maintain the disease-category distribution.

```python
train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
```

---

# 12. Saved Machine Learning Models

The following models are saved:

```text
logistic_regression.pkl
multinomial_naive_bayes.pkl
linear_svm.pkl
```

The label encoder is saved as:

```text
label_encoder.pkl
```

The test data is saved as:

```text
X_test.pkl
y_test.pkl
```

---

# 13. Step 6 - Model Evaluation

File:

```text
step6_evaluation.py
```

Each trained model is evaluated using:

* Accuracy
* Weighted Precision
* Weighted Recall
* Weighted F1-score

### Accuracy

Measures the proportion of correctly classified clinical trials.

### Precision

Measures how many predictions assigned to a category were actually correct.

### Recall

Measures how many actual instances of a category were correctly identified.

### F1-score

Combines precision and recall into a single metric.

### Weighted Metrics

Weighted averaging accounts for the number of samples in each disease category.

---

# 14. Model Selection

The model with the **highest weighted F1-score** is automatically selected.

The selected model is saved as:

```text
selected_model.pkl
```

Model information is stored in:

```text
model_metadata.pkl
```

The complete model comparison is saved as:

```text
model_comparison.csv
```

### Model Comparison Figure

```text
model_performance_comparison.png
```

### Confusion Matrices

Confusion matrices are generated for the trained models to analyze correct and incorrect disease-category predictions.

---

# 15. Step 7 - Insights & Reporting

File:

```text
step7_insights_reporting.py
```

This stage converts the analysis and model results into structured insights.

## Disease Category Distribution

The report identifies:

* Total number of clinical trials
* Number of disease categories
* Number of trials per category
* Percentage distribution of each category

Output:

```text
insights/disease_category_insights.csv
```

---

## Frequently Occurring Medical Terms

The most common terms appearing in the cleaned clinical-trial summaries are reported.

Output:

```text
insights/frequent_medical_terms.csv
```

---

## Clinical Trial Text Patterns

The project analyzes:

* Average summary length
* Median summary length
* Average word count
* Median word count
* Number of disease categories
* Category-level summary patterns

Outputs:

```text
insights/text_pattern_insights.csv
insights/category_text_patterns.csv
```

---

## Model Performance Insights

The performance of all trained models is summarized.

Output:

```text
insights/model_performance_insights.csv
```

A consolidated text report is also generated:

```text
insights/final_insights_report.txt
```

---

# 16. Healthcare Analytics and Research Support

The system can support healthcare analytics and medical research by helping organize and classify large collections of clinical-trial descriptions.

Potential applications include:

* Clinical trial information organization
* Disease-category classification
* Clinical research data exploration
* Medical literature and trial information retrieval
* Trial dataset analysis
* Automated text categorization
* Research-oriented analytics
* Identification of common medical terms and text patterns

The system is intended as an **analytical support tool**. Its predictions should not replace qualified medical, clinical, or research judgment.

---

# 17. Step 8 - Streamlit Application

File:

```text
app.py
```

The project includes an interactive Streamlit web application.

### Main Application Sections

```text
Overview
EDA
Disease Prediction
Model Performance
```

---

## Overview

The Overview page displays information such as:

* Total clinical trials
* Number of disease categories
* Number of TF-IDF features
* Selected machine learning model

---

## EDA

The application displays the generated exploratory analysis, including:

* Disease-category distribution
* Frequently occurring terms
* Clinical-trial text patterns
* EDA visualizations

---

## Disease Prediction

Users can enter a new clinical-trial brief summary.

Example:

```text
A clinical trial will evaluate the safety and effectiveness
of a new treatment in patients with heart failure. The study
will measure cardiovascular function, symptoms, and
hospitalization outcomes.
```

The application performs:

```text
Input Text
    ↓
Text Preprocessing
    ↓
TF-IDF Transformation
    ↓
Selected ML Model
    ↓
Disease Category Prediction
```

The application displays the predicted disease category.

---

## Model Performance

The application displays the performance comparison of:

* Logistic Regression
* Multinomial Naive Bayes
* Linear SVM

Performance metrics include:

* Accuracy
* Weighted Precision
* Weighted Recall
* Weighted F1-score

Interactive charts are created using Plotly.

---

# 18. Project Folder Structure

```text
Clinical_Trial_Disease_Classification/
│
├── clinical_trials_raw_patient2trial_conditions_new.csv
├── clinical_trials_selected.csv
├── clinical_trials_cleaned.csv
│
├── step1_data_collection.py
├── step2_preprocessing.py
├── step3_eda.py
├── step4_tfidf.py
├── step5_train_models.py
├── step6_evaluation.py
├── step7_insights_reporting.py
├── text_preprocessing.py
├── app.py
│
├── tfidf_vectorizer.pkl
├── label_encoder.pkl
├── logistic_regression.pkl
├── multinomial_naive_bayes.pkl
├── linear_svm.pkl
├── selected_model.pkl
├── model_metadata.pkl
├── X_test.pkl
├── y_test.pkl
│
├── model_comparison.csv
├── top_terms_overall.csv
├── top_tfidf_features.csv
├── disease_category_distribution.csv
├── average_summary_length_by_category.csv
├── eda_summary.csv
│
├── model_performance_comparison.png
│
├── figures/
│   ├── category_distribution.png
│   ├── summary_length_distribution.png
│   ├── top_terms_overall.png
│   ├── top_terms_by_category.png
│   ├── summary_length_by_category.png
│   └── average_summary_length_by_category.png
│
└── insights/
    ├── disease_category_insights.csv
    ├── frequent_medical_terms.csv
    ├── text_pattern_insights.csv
    ├── category_text_patterns.csv
    ├── model_performance_insights.csv
    └── final_insights_report.txt
```

---

# 19. Installation

It is recommended to create and activate a virtual environment.

### Create Virtual Environment

```bash
py -3.13 -m venv .venv
```

### Activate Virtual Environment

Windows:

```bash
.venv\Scripts\activate
```

### Install Required Libraries

```bash
python -m pip install pandas numpy scikit-learn nltk matplotlib seaborn plotly streamlit joblib
```

---

# 20. NLTK Resources

The preprocessing module downloads the required NLTK resources:

```text
stopwords
wordnet
omw-1.4
```

They can also be downloaded manually:

```python
import nltk

nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")
```

---

# 21. Running the Project

Run the project steps in the following order.

### Step 1

```bash
python step1_data_collection.py
```

Generates:

```text
clinical_trials_selected.csv
```

### Step 2

```bash
python step2_preprocessing.py
```

Generates:

```text
clinical_trials_cleaned.csv
```

### Step 3

```bash
python step3_eda.py
```

Generates EDA CSV files and figures.

### Step 4

```bash
python step4_tfidf.py
```

Generates:

```text
tfidf_vectorizer.pkl
top_tfidf_features.csv
```

### Step 5

```bash
python step5_train_models.py
```

Generates the trained model files and test data.

### Step 6

```bash
python step6_evaluation.py
```

Evaluates the models and selects the model with the highest weighted F1-score.

### Step 7

```bash
python step7_insights_reporting.py
```

Generates the final insights and reporting files.

### Step 8

Run the Streamlit application:

```bash
python -m streamlit run app.py
```

Alternatively:

```bash
streamlit run app.py
```

The application will open in the browser.

---

# 22. Example Prediction

### Input

```text
A clinical trial evaluates a new therapy for patients with
advanced cancer. The study will assess treatment safety,
tumor response, disease progression, and clinical outcomes.
```

### Processing

```text
Clinical Trial Text
        ↓
Lowercase
        ↓
Remove HTML / URLs / Special Characters
        ↓
Stopword Removal
        ↓
Lemmatization
        ↓
TF-IDF Vectorization
        ↓
Selected Machine Learning Model
        ↓
Disease Category
```

The final category is determined by the trained model and the disease categories present in the training dataset.

---

# 23. Model Evaluation Metrics

The project uses the following evaluation metrics:

| Metric             | Purpose                                                    |
| ------------------ | ---------------------------------------------------------- |
| Accuracy           | Overall percentage of correct predictions                  |
| Weighted Precision | Precision weighted by category support                     |
| Weighted Recall    | Recall weighted by category support                        |
| Weighted F1-score  | Combined precision and recall weighted by category support |

The model with the highest weighted F1-score is automatically selected for deployment.

---

# 24. Advantages

* Uses real clinical-trial text data.
* Automates medical text preprocessing.
* Converts text into machine-learning-ready numerical features.
* Supports multiple classification algorithms.
* Uses stratified train-test splitting.
* Compares models using multiple evaluation metrics.
* Automatically selects the model based on weighted F1-score.
* Provides reusable preprocessing logic.
* Generates structured insights and reports.
* Includes an interactive Streamlit interface.
* Can process new clinical-trial summaries for disease-category prediction.

---

# 25. Limitations

* Classification performance depends on the quality and coverage of the training dataset.
* TF-IDF primarily captures statistical relationships between terms rather than deeper medical context.
* Rare disease categories may have fewer training examples.
* The model may not correctly classify clinical descriptions containing terminology that differs significantly from the training data.
* Predictions should be considered analytical outputs rather than medical diagnoses.
* The system does not replace clinical experts or qualified medical researchers.

---

# 26. Future Enhancements

Possible future improvements include:

* Using domain-specific medical NLP models such as BioBERT or ClinicalBERT.
* Using transformer-based text classification.
* Hyperparameter tuning.
* Cross-validation.
* Handling class imbalance using additional techniques.
* Adding explainable AI features.
* Displaying important terms contributing to predictions.
* Adding confidence calibration.
* Supporting multilingual clinical text.
* Adding a database for storing prediction history.
* Deploying the Streamlit application to a cloud platform.
* Integrating additional clinical-trial metadata.

---

# 27. Project Outcome

The completed system provides an end-to-end NLP and machine-learning pipeline for clinical trial disease-category classification.

The project demonstrates the complete workflow:

```text
Data Collection
      ↓
Data Preprocessing
      ↓
Exploratory Data Analysis
      ↓
NLP Feature Engineering
      ↓
Machine Learning
      ↓
Model Evaluation
      ↓
Insights & Reporting
      ↓
Prediction
      ↓
Streamlit Deployment
```

The final application provides a practical interface for exploring clinical-trial data and performing automated disease-category classification.

---

# 28. Disclaimer

This project is developed for **educational, analytical, and research purposes**.

The predicted disease categories are generated by a machine-learning model and should not be interpreted as medical diagnoses or clinical recommendations.

Clinical, healthcare, and research decisions should be made using appropriate expert review and validated medical information.
