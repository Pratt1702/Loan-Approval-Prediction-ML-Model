from fastapi import FastAPI
from pydantic import BaseModel
from typing import Literal
import pandas as pd
import joblib

app = FastAPI()

# Load pre-trained models and scalers
models = {
    "LogisticRegression": joblib.load("LogisticRegression.joblib"),
    "RandomForestClassifier": joblib.load("RandomForestClassifier.joblib"),
    "SVM": joblib.load("SVM.joblib")
}

scalers = {
    "LogisticRegression": joblib.load("LogisticRegression_scaler.joblib"),
    "RandomForestClassifier": joblib.load("RandomForestClassifier_scaler.joblib"),
    "SVM": joblib.load("SVM_scaler.joblib")
}

# FastAPI Input Schema
class LoanInput(BaseModel):
    Married: int
    Dependents: int
    Education: int
    ApplicantIncome: int
    CoapplicantIncome: float
    LoanAmount: float
    Loan_Amount_Term: float
    Credit_History: float
    Property_Area: int
    use_smote: bool
    model_name: Literal["LogisticRegression", "RandomForestClassifier", "SVM"]

class LoanOutput(BaseModel):
    approved: bool

# FastAPI Endpoint
@app.post("/predict", response_model=LoanOutput)
def predict(input: LoanInput):
    input_dict = input.dict()
    model_name = input_dict.pop("model_name")
    
    # Prepare input for prediction
    input_df = pd.DataFrame([input_dict])
    scaler = scalers[model_name]
    input_scaled = scaler.transform(input_df)
    
    # Make prediction
    model = models[model_name]
    pred = model.predict(input_scaled)
    
    return {"approved": bool(pred[0])}