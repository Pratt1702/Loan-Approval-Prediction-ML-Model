from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware  # Import CORSMiddleware
from pydantic import BaseModel
from typing import Literal
import pandas as pd
import joblib

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173","https://loan-approval-prediction-six.vercel.app"],  # Adjust the port if necessary
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    model_name = input_dict.pop("model_name")  # remove model_name
    input_dict.pop("use_smote", None)          # also remove use_smote
    
    # Prepare input for prediction
    input_df = pd.DataFrame([input_dict])
    scaler = scalers[model_name]
    input_scaled = scaler.transform(input_df)
    
    # Make prediction
    model = models[model_name]
    pred = model.predict(input_scaled)
    
    return {"approved": bool(pred[0])}

@app.get("/health")
def health_check():
    return {"status": "ok"}