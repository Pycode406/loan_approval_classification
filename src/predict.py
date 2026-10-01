import os
import joblib
import pandas as pd
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "loan_approval_model.pkl"
)
model = joblib.load(MODEL_PATH)
while True:
    print("\n--- Loan Approval Prediction ---")
    print("Enter 'exit' to stop.")
    age_input = input("Enter Age: ")
    if age_input.lower() == "exit":
        break
    income_input = input("Enter Income: ")
    loan_input = input("Enter Loan Amount: ")
    credit_input = input("Enter Credit Score: ")
    employment_type = input(
        "Enter Employment Type (Salaried/Self-Employed): "
    )
    try:
        age = int(age_input)
        income = float(income_input)
        loan_amount = float(loan_input)
        credit_score = float(credit_input)
    except ValueError:
        print("\nPlease enter valid numeric values.")
        continue
    if age <= 0:
        print("\nAge must be greater than 0.")
        continue
    if income < 0:
        print("\nIncome cannot be negative.")
        continue
    if loan_amount < 0:
        print("\nLoan amount cannot be negative.")
        continue
    if credit_score < 0 or credit_score > 900:
        print("\nCredit score must be between 0 and 900.")
        continue
    if employment_type not in [
        "Salaried",
        "Self-Employed"
    ]:
        print(
            "\nEmployment Type must be "
            "Salaried or Self-Employed."
        )
        continue
    new_record = pd.DataFrame([{
        "Age": age,
        "Income": income,
        "Loan_Amount": loan_amount,
        "Credit_Score": credit_score,
        "Employment_Type": employment_type
    }])

    prediction = model.predict(new_record)[0]
    probability = model.predict_proba(new_record)[0][1]
    print("\n-----------------------------")

    if prediction == 1:
        print("Prediction: Loan Approved")
    else:
        print("Prediction: Loan Not Approved")
    print(
        f"Approval Probability: "
        f"{probability * 100:.2f}%"
    )
    print("-----------------------------")