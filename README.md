# 🧬 Clinical Trial Disease Category Classification Using NLP

## 📌 Project Overview

The **Clinical Trial Disease Category Classification Using NLP** project is an end-to-end Natural Language Processing and Machine Learning application designed to classify clinical trial summaries into appropriate disease categories.

Clinical trial datasets contain large amounts of unstructured medical text, including study summaries, treatment information, eligibility criteria, disease information, and clinical outcomes. Manually analyzing this information can be difficult and time-consuming.

This project focuses on the **Brief Summary** of each clinical trial and applies **Natural Language Processing (NLP)** and **Machine Learning** techniques to automatically predict the corresponding **Disease Category**.

The final solution includes an interactive **Streamlit web application** for:

- Clinical trial data exploration
- Exploratory Data Analysis (EDA)
- Disease category prediction
- Model performance comparison
- Medical text pattern analysis

---

# 🎯 Project Objectives

The main objectives of this project are:

- Collect and understand clinical trial data
- Clean and preprocess clinical trial summaries
- Perform Exploratory Data Analysis on medical text
- Apply NLP techniques to clinical trial summaries
- Convert medical text into numerical features using TF-IDF
- Train multiple Machine Learning classification models
- Compare model performance using classification metrics
- Select the model using Macro F1 Score
- Predict disease categories from new clinical trial summaries
- Analyze disease-specific patterns and frequently occurring medical terms
- Develop an interactive Streamlit application
- Build an end-to-end medical NLP classification pipeline

---

# 📊 Dataset

The project uses a clinical trial dataset containing **60,337 clinical trial records**.

The original dataset contains **16 columns** with information such as:

- Clinical trial ID
- Trial title
- Official title
- Brief summary
- Conditions
- Interventions
- Study status
- Study type
- Phase
- Sex
- Minimum age
- Maximum age
- Healthy volunteer information
- Eligibility criteria
- ClinicalTrials URL
- Disease category

For this classification problem, the main columns used are:

| Column | Purpose |
|---|---|
| `nct_id` | Unique clinical trial identifier |
| `brief_summary` | Input medical text used for NLP |
| `source_condition_query` | Original disease-category target |
| `disease_category` | Renamed target variable used for classification |

---

# 🏷️ Disease Categories

The dataset contains **8 disease categories**:

1. Breast Cancer
2. Type 2 Diabetes
3. COVID-19
4. Anxiety
5. Chronic Obstructive Pulmonary Disease
6. Rheumatoid Arthritis
7. Glaucoma
8. Sickle Cell Anemia

The dataset is **imbalanced**, meaning some disease categories contain more clinical trials than others.

Therefore, both **Weighted** and **Macro** evaluation metrics are used during model evaluation.

---

# 🔄 Project Workflow

```text
Clinical Trial Dataset
        ↓
Data Collection
        ↓
Data Preprocessing
        ↓
Exploratory Data Analysis
        ↓
NLP Text Processing
        ↓
TF-IDF Feature Extraction
        ↓
Train-Test Split
        ↓
Machine Learning Model Training
        ↓
Model Evaluation
        ↓
Best Model Selection
        ↓
Insights & Reporting
        ↓
Disease Category Prediction
        ↓
Streamlit Application
```

---

# 1️⃣ Data Collection

The first stage loads and analyzes the raw clinical trial dataset.

The following operations are performed:

- Load the clinical trial CSV dataset
- Check dataset shape
- Display column names
- Inspect data types
- Analyze missing values
- Check duplicate clinical trial IDs
- Analyze the `brief_summary` column
- Analyze disease-category distribution
- Select the required columns
- Rename the target column to `disease_category`

Selected columns:

```python
nct_id
brief_summary
disease_category
```

The selected dataset is saved as:

```text
clinical_trials_selected.csv
```

---

# 2️⃣ Data Preprocessing

Clinical trial summaries contain unstructured medical text.

The preprocessing stage prepares this text for Machine Learning.

### Text preprocessing includes:

- Converting text to lowercase
- Removing HTML tags
- Removing URLs
- Removing special characters
- Removing unnecessary spaces
- Tokenizing text
- Removing English stopwords
- Preserving important negation words:
  - `no`
  - `not`
  - `nor`
- Applying WordNet lemmatization

The project also calculates:

- Original text length
- Cleaned text length
- Word count

The cleaned dataset is saved as:

```text
clinical_trials_cleaned.csv
```

---

# 3️⃣ Exploratory Data Analysis

Exploratory Data Analysis is performed to understand the structure and patterns of clinical trial data.

### EDA includes:

- Disease category distribution
- Clinical trial count by disease category
- Clinical summary length distribution
- Most frequently occurring medical terms
- Top medical terms for each disease category
- Average summary length by disease category
- Category-level text patterns
- Medical text statistics

Visualizations are generated using:

- Matplotlib
- Seaborn
- Plotly

---

# 4️⃣ NLP and TF-IDF Feature Extraction

Machine Learning models cannot directly process raw text.

Therefore, clinical trial summaries are transformed into numerical features using:

## TF-IDF

**TF-IDF** stands for:

> Term Frequency – Inverse Document Frequency

TF-IDF measures the importance of words and phrases within documents relative to the complete collection of clinical trial summaries.

The TF-IDF vectorizer uses:

```python
max_features=20000
min_df=2
max_df=0.95
ngram_range=(1, 2)
sublinear_tf=True
strip_accents="unicode"
```

Both:

- Unigrams
- Bigrams

are included.

The trained vectorizer is saved as:

```text
tfidf_vectorizer.pkl
```

---

# 5️⃣ Train-Test Split

To prevent test-data leakage, the dataset is split **before fitting the final TF-IDF vectorizer**.

The workflow is:

```text
Cleaned Text
      ↓
Train-Test Split
      ↓
Fit TF-IDF on Training Data
      ↓
Transform Training Data
      ↓
Transform Test Data
```

The split uses:

```python
test_size=0.20
random_state=42
stratify=y
```

Stratification helps preserve the disease-category distribution in both training and testing datasets.

---

# 6️⃣ Machine Learning Models

Three Machine Learning classification algorithms are trained.

## Logistic Regression

A linear classification algorithm suitable for high-dimensional sparse text features.

## Multinomial Naive Bayes

A probabilistic classification algorithm commonly used for NLP and text-classification problems.

## Linear Support Vector Machine

Linear SVM is effective for high-dimensional text classification and identifies decision boundaries between disease categories.

The trained models are saved as:

```text
logistic_regression.pkl
multinomial_naive_bayes.pkl
linear_svm.pkl
```

---

# 7️⃣ Model Evaluation

Each Machine Learning model is evaluated using multiple classification metrics.

### Evaluation Metrics

- Accuracy
- Weighted Precision
- Weighted Recall
- Weighted F1 Score
- Macro Precision
- Macro Recall
- Macro F1 Score
- Classification Report
- Confusion Matrix

### Why Macro F1?

The clinical trial dataset contains imbalanced disease categories.

**Macro F1 Score** calculates the F1 score independently for every disease category and then gives equal importance to each class.

Therefore, Macro F1 is used as the primary metric for selecting the final model.

The model comparison results are saved as:

```text
model_comparison.csv
```

The selected model is saved as:

```text
selected_model.pkl
```

Model information and evaluation metrics are stored in:

```text
model_metadata.pkl
```

---

# 8️⃣ Insights and Reporting

The project generates insights about clinical trial data and model performance.

The reporting stage includes:

### Disease Category Distribution

Analyzes how clinical trials are distributed across disease categories.

### Frequently Occurring Medical Terms

Identifies commonly occurring terms within clinical trial summaries.

### Clinical Trial Text Patterns

Analyzes:

- Summary lengths
- Word counts
- Average text length
- Median text length

### Disease-Specific Patterns

Compares text characteristics across different disease categories.

### Model Performance

Reports:

- Accuracy
- Precision
- Recall
- F1 Score
- Macro metrics
- Weighted metrics

These insights support healthcare analytics and medical research exploration.

---

# 9️⃣ Disease Category Prediction

The prediction pipeline accepts a new clinical trial brief summary.

The prediction workflow is:

```text
New Clinical Trial Summary
        ↓
Text Preprocessing
        ↓
TF-IDF Transformation
        ↓
Selected ML Model
        ↓
Encoded Prediction
        ↓
Label Encoder
        ↓
Predicted Disease Category
```

The model only predicts disease categories that were included in the training dataset.

---

# 🖥️ Streamlit Application

An interactive Streamlit application is developed to demonstrate the complete project.

The application contains four main pages.

## 📊 Overview

Displays:

- Number of clinical trials
- Number of disease categories
- Number of TF-IDF features
- Selected Machine Learning model
- Project objective
- Machine Learning pipeline
- Models used
- Supported disease categories

## 📈 EDA

Displays:

- Disease category distribution
- Summary length distribution
- Important medical terms
- Average summary length by disease category
- Dataset preview

## 🧠 Disease Prediction

The user can enter or paste a clinical trial brief summary.

The application then:

1. Cleans the medical text
2. Converts the text into TF-IDF features
3. Applies the selected Machine Learning model
4. Predicts the disease category
5. Displays the predicted disease category
6. Displays prediction probabilities when supported by the model
7. Displays decision scores when Linear SVM is selected

## 📊 Model Performance

Displays:

- Model comparison table
- Accuracy
- Weighted Precision
- Weighted Recall
- Weighted F1
- Macro Precision
- Macro Recall
- Macro F1
- Selected model
- Performance comparison visualization

---

# 📁 Project Structure

```text
Clinical_Trial_Disease_Classification/
│
├── Step1_Data_Collection.py
├── Step2_Data_Preprocessing.py
├── Step3_EDA.py
├── Step4_TFIDF.py
├── Step5_Model_Training.py
├── Step6_Model_Evaluation.py
├── Step7_Insights_Reporting.py
├── Step8_Disease_Prediction.py
│
├── app.py
├── README.md
├── requirements.txt
│
├── clinical_trials_selected.csv
├── clinical_trials_cleaned.csv
│
├── tfidf_vectorizer.pkl
├── label_encoder.pkl
├── logistic_regression.pkl
├── multinomial_naive_bayes.pkl
├── linear_svm.pkl
├── selected_model.pkl
├── model_metadata.pkl
│
├── X_test.pkl
├── y_test.pkl
├── model_comparison.csv
├── top_terms_overall.csv
│
├── figures/
│
└── insights/
```

---

# 🛠️ Technologies Used

## Programming Language

- Python

## Data Analysis

- Pandas
- NumPy

## Natural Language Processing

- NLTK
- TF-IDF
- Stopword Removal
- Lemmatization
- Unigrams
- Bigrams

## Machine Learning

- Scikit-learn

## Machine Learning Algorithms

- Logistic Regression
- Multinomial Naive Bayes
- Linear SVM

## Data Visualization

- Matplotlib
- Seaborn
- Plotly

## Model Storage

- Joblib

## Web Application

- Streamlit

---

# 📦 Required Python Libraries

Install the required libraries using:

```bash
pip install pandas numpy scikit-learn nltk matplotlib seaborn plotly streamlit joblib
```

Alternatively, install from `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

# 📄 requirements.txt

A basic `requirements.txt` can contain:

```text
pandas
numpy
scikit-learn
nltk
matplotlib
seaborn
plotly
streamlit
joblib
```

---

# ▶️ How to Run the Project

## Step 1: Create Virtual Environment

```bash
python -m venv .venv
```

## Step 2: Activate Virtual Environment

### Windows

```bash
.venv\Scripts\activate
```

## Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 4: Run the Project Pipeline

Run the Python scripts in sequence:

```bash
python Step1_Data_Collection.py
python Step2_Data_Preprocessing.py
python Step3_EDA.py
python Step4_TFIDF.py
python Step5_Model_Training.py
python Step6_Model_Evaluation.py
python Step7_Insights_Reporting.py
```

## Step 5: Run Streamlit Application

```bash
streamlit run app.py
```

The Streamlit application will open in the web browser.

---

# 📌 Model Artifacts

The application requires the following trained files:

```text
tfidf_vectorizer.pkl
selected_model.pkl
label_encoder.pkl
model_metadata.pkl
```

These files must be available in the same project directory as `app.py`.

---

# ⚠️ Important Model Limitation

This project is a **closed-set multiclass classification system**.

The trained model can only classify a clinical trial into one of the disease categories present in the training dataset.

If a clinical trial summary belongs to a completely different disease that was not included during training, the model will still assign it to one of the known categories.

Therefore, predictions should always be interpreted within the scope of the training dataset.

---

# ⚕️ Medical Disclaimer

This project is developed for **educational, academic, research, and data-science demonstration purposes only**.

The predictions generated by the Machine Learning model:

- Are not medical diagnoses
- Should not be used for patient care
- Should not be used for treatment decisions
- Should not replace healthcare professionals
- Should not be used for clinical decision-making

The system demonstrates the application of NLP and Machine Learning techniques to clinical trial text classification.

---

# 💡 Key Learning Outcomes

This project demonstrates practical knowledge of:

- Data collection
- Data preprocessing
- Exploratory Data Analysis
- Natural Language Processing
- Text cleaning
- Stopword removal
- Lemmatization
- TF-IDF vectorization
- Feature extraction
- Multiclass classification
- Class imbalance evaluation
- Machine Learning model comparison
- Accuracy, Precision, Recall and F1 Score
- Macro and Weighted evaluation metrics
- Confusion Matrix
- Model serialization
- New-text prediction
- Streamlit application development
- End-to-end Machine Learning project implementation

---

# 🚀 Future Enhancements

Possible future improvements include:

- Adding more disease categories
- Expanding the clinical trial dataset
- Comparing TF-IDF models with word embeddings
- Exploring transformer-based NLP models such as BERT
- Adding confidence-threshold handling for uncertain predictions
- Improving handling of clinical trials outside the trained disease categories
- Adding additional clinical trial features
- Deploying the Streamlit application to a cloud platform
- Developing advanced medical-text analytics dashboards

---

# ✅ Conclusion

The **Clinical Trial Disease Category Classification Using NLP** project demonstrates an end-to-end Machine Learning workflow for automatically classifying clinical trial brief summaries into disease categories.

The project combines:

**Clinical Trial Data + Text Preprocessing + NLP + TF-IDF + Machine Learning + Model Evaluation + Streamlit**

to create an interactive medical-text classification system.

The final application allows users to explore clinical trial data, analyze disease-specific patterns, compare Machine Learning models, and classify new clinical trial summaries into supported disease categories.

---

## 👨‍💻 Project Type

**Data Science | Natural Language Processing | Machine Learning | Healthcare Analytics**

## 📚 Use

**Educational and Research Purposes Only**
