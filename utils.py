
import joblib
import numpy as np

model = joblib.load("loan_eligibility_model.pkl")
encoders = joblib.load("encoders.pkl")  
def preprocess_input(data):
    data['Gender'] = encoders['Gender'].transform([data['Gender']])[0]
    data['Married'] = encoders['Married'].transform([data['Married']])[0]
    data['Education'] = encoders['Education'].transform([data['Education']])[0]
    data['Self_Employed'] = encoders['Self_Employed'].transform([data['Self_Employed']])[0]
    data['Property_Area'] = encoders['Property_Area'].transform([data['Property_Area']])[0]
    data['Dependents'] = encoders['Dependents'].transform([data['Dependents']])[0]

    features = [
        data['Gender'], data['Married'], data['Dependents'], data['Education'],
        data['Self_Employed'], data['ApplicantIncome'], data['CoapplicantIncome'],
        data['LoanAmount'], data['Loan_Amount_Term'], data['Credit_History'],
        data['Property_Area']
    ]
    return np.array([features])

def predict_eligibility(input_dict):
    X = preprocess_input(input_dict)
    pred = model.predict(X)[0]
    return "Eligible" if pred == 1 else "Not Eligible"
