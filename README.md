# Loan Approval Prediction Project

This project demonstrates a complete machine learning pipeline focused on predicting loan approval. The pipeline is divided into three main parts:

1. **Data Generation:** Create 10,000 synthetic loan application records using concurrent programming.
2. **Model Training:** Train a Logistic Regression model on the generated data and save the model along with its preprocessing objects.
3. **Prediction Interface:** Load the saved model and preprocessing objects to predict loan approval for new user inputs via a command-line interface.

The project is designed for beginners (freshers) who want to build a foundation in applying machine learning techniques in a financial context.

---

## Table of Contents

- [Overview](#overview)
- [Data Requirements](#data-requirements)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
  - [Generate Synthetic Data](#generate-synthetic-data)
  - [Train the Model](#train-the-model)
  - [Run Predictions](#run-predictions)
- [Removing Cached Git Credentials (Optional)](#removing-cached-git-credentials-optional)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

This project creates a synthetic dataset of loan applications and builds a machine learning model to predict whether a loan application will be approved. The steps include:
- **Data Generation:** Using the Faker library and Python's `concurrent.futures`, 10,000 fake loan records are generated and saved in a CSV file.
- **Model Training:** Data is preprocessed (missing values, label encoding for categorical variables, and scaling of numerical features) before training a Logistic Regression model. The model, scaler, and encoders are saved for later use.
- **Prediction Interface:** A command-line interface prompts users to enter loan application details. The input is preprocessed in the same manner as the training data, then the trained model predicts the loan approval status and outputs the probability.

---

## Data Requirements

The synthetic dataset simulates real-world loan application data with the following fields:
- **Loan_ID:** Unique identifier for the loan.
- **Gender:** Applicant's gender ("Male" or "Female").
- **Married:** Marital status ("Yes" or "No").
- **Dependents:** Number of dependents (e.g., "0", "1", "2", "3+").
- **ApplicantIncome:** Applicant's income.
- **CoapplicantIncome:** Co-applicant's income.
- **LoanAmount:** Requested loan amount (in thousands).
- **Loan_Amount_Term:** Loan term in months.
- **Credit_History:** Credit history (1.0 for good, 0.0 for bad).
- **Property_Area:** Area where the property is located ("Urban", "Semiurban", or "Rural").
- **Loan_Status:** Target variable for model training ("Y" for approved, "N" for rejected).

---

## Project Structure
```
Essential-Data-Requirements:
├── predict.py
├── train.py
├── loan_model.pkl
├── loan_applications.csv
├── scaler.pkl
├── encoder.pkl
├── README.md 
└── requirements.txt 
```
---

## Setup and Installation

1. **Clone the Repository or Create Your Project Folder:**

```
   git clone "https://github.com/Saumyasundaram23/Essential-Data-Requirements.git"
   cd Essential-Data-Requirements
```
(Optional) Create a Virtual Environment:

```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
Install Required Dependencies:
Ensure you have a requirements.txt file with the following (or similar) content:
```
nginx
pandas
numpy
scikit-learn
Faker
joblib
```
Then run:
```
pip install -r requirements.txt
```
Usage
This project is controlled via a single Python script (project.py) that supports three modes: generate, train, and predict.

Generate Synthetic Data
Run this command to generate 10,000 fake loan application records and save them as loan_applications.csv:

python project.py generate
Train the Model
Once data is generated, train the model by executing:

```bash
python project.py train
This will:
```
Load the data from loan_applications.csv.
Preprocess the data (forward-fill missing values, label-encode categorical variables, scale numerical features).
Train a Logistic Regression model on the full dataset.
Save the trained model (loan_model.pkl), scaler (scaler.pkl), and encoders (encoders.pkl).
Run Predictions
To predict loan approval for a new loan application, run:

```
python project.py predict
```
You will be prompted to enter details (e.g., Gender, Married, Applicant Income, etc.). The script will:


License
This project is licensed under the MIT License. See the LICENSE file for details.

Additional Information
This project demonstrates a complete machine learning workflow with an emphasis on reproducibility and ease of use. It is designed for educational purposes and to build foundational skills in both machine learning and financial risk analysis.

Happy coding and exploring!

---

This README file now provides a single, unified overview of the project along with clear instructions on how to generate data, train the model, and run predictions—all in one file. Adjust paths and instructions as needed for your specific environment.







