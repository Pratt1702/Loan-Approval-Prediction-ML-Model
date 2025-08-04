from fastapi import FastAPI
from pydantic import BaseModel
from typing import Literal
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE

app = FastAPI()

# --- Data Loading and Preprocessing ---

def load_and_preprocess_data():
    df_train = pd.read_csv("train_u6lujuX_CVtuZ9i.csv")
    cols = ['Gender', 'Married', 'Dependents', 'Self_Employed', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']
    for col in cols:
        df_train[col] = df_train[col].fillna(df_train[col].mode()[0])
    le = LabelEncoder()
    cat_cols = ['Gender', 'Married', 'Education', 'Property_Area', 'Dependents', 'Self_Employed']
    for col in cat_cols:
        df_train[col] = le.fit_transform(df_train[col])
    df_train['Loan_Status'] = df_train['Loan_Status'].map({'Y': 1, 'N': 0})
    return df_train

def get_features_and_target(df):
    X = df.drop(['Loan_ID', 'Loan_Status', 'Gender', 'Self_Employed'], axis=1)
    y = df['Loan_Status']
    return X, y

def scale_and_smote(X, y):
    scaler = StandardScaler()
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train_scaled = scaler.fit_transform(X_train)
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
    return X_train_resampled, y_train_resampled, scaler

def scale_only(X, y):
    scaler = StandardScaler()
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train_scaled = scaler.fit_transform(X_train)
    return X_train_scaled, y_train, scaler

def get_model(model_name):
    if model_name == "LogisticRegression":
        return LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    elif model_name == "RandomForestClassifier":
        return RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_name == "SVM":
        return SVC(random_state=42, probability=True)
    else:
        raise ValueError("Invalid model name. Choose from 'LogisticRegression', 'RandomForestClassifier', 'SVM'.")

def predict_loan_approval(input_dict, use_smote, model_name):
    df = load_and_preprocess_data()
    X, y = get_features_and_target(df)
    if use_smote:
        X_train_scaled, y_train, scaler = scale_and_smote(X, y)
    else:
        X_train_scaled, y_train, scaler = scale_only(X, y)
    model = get_model(model_name)
    model.fit(X_train_scaled, y_train)
    input_df = pd.DataFrame([input_dict])
    input_scaled = scaler.transform(input_df)
    pred = model.predict(input_scaled)
    return bool(pred[0])

# --- FastAPI Input Schema ---

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

# --- FastAPI Endpoint ---

@app.post("/predict", response_model=LoanOutput)
def predict(input: LoanInput):
    input_dict = input.dict()
    use_smote = input_dict.pop("use_smote")
    model_name = input_dict.pop("model_name")
    approved = predict_loan_approval(input_dict, use_smote, model_name)
    return {"approved": approved}