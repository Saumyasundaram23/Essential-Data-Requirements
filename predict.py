import pandas as pd
import joblib

# File paths (should match those used in the training script)
MODEL_FILE = "loan_model.pkl"
SCALER_FILE = "scaler.pkl"
ENCODERS_FILE = "encoders.pkl"

# Load the saved model, scaler, and encoders
model = joblib.load(MODEL_FILE)
scaler = joblib.load(SCALER_FILE)
encoders = joblib.load(ENCODERS_FILE)

# Define the feature order used in training
features = ['Gender', 'Married', 'Dependents', 'ApplicantIncome', 
            'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 
            'Credit_History', 'Property_Area']

def get_user_input():
    print("Enter details for the loan application:")
    
    # Loop until valid input is given for Gender
    while True:
        gender = input("Gender (Male/Female): ").strip().capitalize()
        if gender in ["Male", "Female"]:
            break
        else:
            print("Invalid input for Gender. Please enter either 'Male' or 'Female'.")
    
    # For Married, we can also ensure the user enters "Yes" or "No"
    while True:
        married = input("Married (Yes/No): ").strip().capitalize()
        if married in ["Yes", "No"]:
            break
        else:
            print("Invalid input for Married. Please enter 'Yes' or 'No'.")
    
    dependents = input("Dependents (0, 1, 2, 3+): ").strip()
    
    try:
        applicant_income = float(input("Applicant Income: ").strip())
        coapplicant_income = float(input("Coapplicant Income: ").strip())
        loan_amount = float(input("Loan Amount (in thousands): ").strip())
        loan_amount_term = float(input("Loan Amount Term (in months): ").strip())
        credit_history = float(input("Credit History (1.0 for good, 0.0 for bad): ").strip())
    except ValueError:
        print("Invalid input for numerical value. Please enter numbers only.")
        exit(1)
    
    while True:
        property_area = input("Property Area (Urban, Semiurban, Rural): ").strip().capitalize()
        if property_area in ["Urban", "Semiurban", "Rural"]:
            break
        else:
            print("Invalid input for Property Area. Please enter Urban, Semiurban, or Rural.")
    
    new_data = {
        "Gender": gender,
        "Married": married,
        "Dependents": dependents,
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_amount_term,
        "Credit_History": credit_history,
        "Property_Area": property_area
    }
    return new_data

# The rest of the script remains the same.

def main():
    # Get user input from command line
    user_input = get_user_input()
    
    # Convert the input into a DataFrame
    new_df = pd.DataFrame([user_input])
    
    # Preprocess categorical variables using saved encoders
    categorical_cols = list(encoders.keys())
    for col in categorical_cols:
        try:
            new_df[col] = encoders[col].transform(new_df[col])
        except Exception as e:
            print(f"Error encoding column {col}: {e}")
            exit(1)
    
    # Scale the numerical features using the saved scaler
    numerical_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 
                      'Loan_Amount_Term', 'Credit_History']
    try:
        new_df[numerical_cols] = scaler.transform(new_df[numerical_cols])
    except Exception as e:
        print("Error scaling numerical features:", e)
        exit(1)
    
    # Ensure that the DataFrame contains features in the correct order
    X_new = new_df[features]
    
    # Predict using the loaded model
    predicted_class = model.predict(X_new)[0]
    predicted_prob = model.predict_proba(X_new)[0][1]  # probability for the approved class
    
    # Output the prediction result
    status = "Approved" if predicted_class == 1 else "Rejected"
    print("\nPrediction Result:")
    print(f"Loan Approval Status: {status}")
    print(f"Approval Probability: {predicted_prob*100:.2f}%")
    
if __name__ == "__main__":
    main()
