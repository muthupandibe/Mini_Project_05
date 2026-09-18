# STEP 1: DATA COLLECTION

import pandas as pd

# ---------------------------------------------------------------------
# 1. Import the dataset
# ---------------------------------------------------------------------
INPUT_PATH = "clinical_trials_raw_patient2trial_conditions_new.csv"  # update path as needed

df = pd.read_csv(INPUT_PATH)

# ---------------------------------------------------------------------
# 2. Understand the dataset structure, columns, and data types
# ---------------------------------------------------------------------
print("Shape (rows, columns):", df.shape)
print("\nColumn names:\n", df.columns.tolist())
print("\nData types:\n", df.dtypes)
print("\nFirst 5 rows:\n", df.head())
print("\nMissing values per column:\n", df.isna().sum())
print("\nDuplicate trial IDs (nct_id):", df.duplicated(subset=["nct_id"]).sum())

# ---------------------------------------------------------------------
# 3. Select the "Brief Summary" column for NLP-based analysis
# ---------------------------------------------------------------------
brief_summary = df["brief_summary"]
print("\nSample brief_summary text:\n", brief_summary.iloc[0])
print("\nAverage summary length (characters):", brief_summary.str.len().mean())

# ---------------------------------------------------------------------
# 4. Identify disease category labels for classification
#    (source_condition_query holds the disease-category label for each trial)
# ---------------------------------------------------------------------
disease_labels = df["source_condition_query"]
print("\nUnique disease categories:", disease_labels.nunique())
print("\nDisease category distribution:\n", disease_labels.value_counts())

# ---------------------------------------------------------------------
# 5. Keep only the columns needed downstream and rename for clarity
# ---------------------------------------------------------------------
df_selected = df[["nct_id", "brief_summary", "source_condition_query"]].rename(
    columns={"source_condition_query": "disease_category"}
)
print("\nSelected columns for the pipeline:\n", df_selected.head())

if __name__ == "__main__":
    # Save the selected subset so later steps can load it directly
    df_selected.to_csv("clinical_trials_selected.csv", index=False)
    print("\nSaved selected columns to clinical_trials_selected.csv")