import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib

# File paths
DATA_FILE = "loan_applications.csv"
MODEL_FILE = "loan_model.pkl"
SCALER_FILE = "scaler.pkl"
ENCODERS_FILE = "encoders.pkl"

# Load data
data = pd.read_csv(DATA_FILE)

# Fill missing values (if any)
data.ffill(inplace=True)


# Define features and target
features = ['Gender', 'Married', 'Dependents', 'ApplicantIncome', 
            'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 
            'Credit_History', 'Property_Area']
target = 'Loan_Status'

# Prepare a dictionary for LabelEncoders for categorical columns
categorical_cols = ['Gender', 'Married', 'Dependents', 'Property_Area']
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    encoders[col] = le

# Scale numerical features
numerical_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 
                  'Loan_Amount_Term', 'Credit_History']
scaler = StandardScaler()
data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

# Separate features and target
X = data[features]
y = data[target]

# Train logistic regression model on entire dataset (for demonstration)
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# Save the trained model, scaler, and encoders using joblib
joblib.dump(model, MODEL_FILE)
joblib.dump(scaler, SCALER_FILE)
joblib.dump(encoders, ENCODERS_FILE)

print(f"Model, scaler, and encoders have been saved to {MODEL_FILE}, {SCALER_FILE} and {ENCODERS_FILE} respectively.")
