import pandas as pd
import numpy as np

# Load dataset
file_path = r"G:\New folder\task\Delinquency_prediction_dataset.xlsx"
df = pd.read_excel(file_path)

print("=" * 50)
print("DATASET OVERVIEW")
print("=" * 50)

print(f"Number of Records: {df.shape[0]}")
print(f"Number of Columns: {df.shape[1]}")

print("\nColumns:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

print("\n" + "=" * 50)
print("MISSING VALUE ANALYSIS")
print("=" * 50)

missing = df.isnull().sum()
missing = missing[missing > 0]

print("\nMissing Values:")
print(missing)

# Median Imputation
for col in df.columns:
    if df[col].isnull().sum() > 0:
        if pd.api.types.is_numeric_dtype(df[col]):
            median_value = df[col].median()
            df[col].fillna(median_value, inplace=True)

print("\nMissing values after imputation:")
print(df.isnull().sum().sum())

print("\n" + "=" * 50)
print("DUPLICATE CHECK")
print("=" * 50)

duplicates = df.duplicated().sum()
print(f"Duplicate Records: {duplicates}")

print("\n" + "=" * 50)
print("NUMERICAL SUMMARY")
print("=" * 50)

print(df.describe())

print("\n" + "=" * 50)
print("CORRELATION ANALYSIS")
print("=" * 50)

numeric_df = df.select_dtypes(include=np.number)

corr_matrix = numeric_df.corr()

print(corr_matrix)

# Delinquency column check
target_cols = [
    col for col in df.columns
    if "delinq" in col.lower()
    or "default" in col.lower()
    or "risk" in col.lower()
]

if target_cols:
    target = target_cols[0]

    print(f"\nTarget Variable Found: {target}")

    target_corr = corr_matrix[target].sort_values(
        ascending=False
    )

    print("\nTop Correlations:")
    print(target_corr)

print("\n" + "=" * 50)
print("HIGH RISK INDICATORS")
print("=" * 50)

risk_indicators = [
    "High Credit Utilization",
    "High Debt-to-Income Ratio",
    "Previous Missed Payments",
    "Low Account Tenure",
    "Employment Instability"
]

for item in risk_indicators:
    print(f"- {item}")

print("\nEDA Completed Successfully.")