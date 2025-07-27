import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

df = pd.read_csv('dataset/loan.csv', low_memory=False)

# Drop columns with more than 30% missing values
missing_data = df.isnull().sum()
missing_percentage = (missing_data / len(df)) * 100
cols_to_drop = missing_percentage[missing_percentage > 30].index
df.drop(columns=cols_to_drop, inplace=True)

# Drop columns with only one unique value (constant columns)
for col in df.columns:
    if df[col].nunique() == 1:
        df.drop(columns=col, inplace=True)

# Convert 'term' and 'emp_length' to numerical
df['term'] = df['term'].apply(lambda x: int(x.split()[0]))
df['emp_length'].fillna('0 years', inplace=True)
df['emp_length'] = df['emp_length'].replace({'< 1 year': '0 years', '10+ years': '10 years'}).apply(lambda x: int(x.split()[0]))

# Convert 'int_rate' and 'revol_util' to numerical
# Check if 'int_rate' is object type before applying str.replace
if df['int_rate'].dtype == 'object':
    df['int_rate'] = df['int_rate'].str.replace('%', '').astype(float)
if df['revol_util'].dtype == 'object':
    df['revol_util'] = df['revol_util'].str.replace('%', '').astype(float)

# Define target variable
# 'loan_status' is the target. We need to map it to binary (default/not default).
# Based on LendingClub data, 'Charged Off', 'Default', 'Late (31-120 days)', 'Does not meet the credit policy. Status:Charged Off' are considered default.
# 'Fully Paid', 'Current', 'In Grace Period', 'Late (16-30 days)', 'Does not meet the credit policy. Status:Fully Paid' are not default.

default_statuses = ['Charged Off', 'Default', 'Late (31-120 days)', 'Does not meet the credit policy. Status:Charged Off']
df['is_default'] = df['loan_status'].apply(lambda x: 1 if x in default_statuses else 0)

# Drop the original 'loan_status' column and other non-useful columns
df.drop(columns=['loan_status', 'zip_code', 'issue_d', 'earliest_cr_line', 'title', 'emp_title', 'last_pymnt_d', 'last_credit_pull_d'], errors='ignore', inplace=True)

# Handle remaining missing values
# For numerical columns, fill with median
for col in df.select_dtypes(include=np.number).columns:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].median(), inplace=True)

# For categorical columns, fill with mode
for col in df.select_dtypes(include='object').columns:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].mode()[0], inplace=True)

# One-hot encode categorical features
categorical_cols = df.select_dtypes(include='object').columns
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Prepare data for modeling
X = df.drop(columns=['is_default'])
y = df['is_default']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Build and train a RandomForestClassifier model
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1) # n_jobs=-1 uses all available cores
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("\nModel Evaluation:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))