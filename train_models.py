import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
import joblib

# Load and preprocess data
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

# Train and save models
def train_and_save_models():
    df = load_and_preprocess_data()
    X, y = get_features_and_target(df)

    # Scale and apply SMOTE
    scaler = StandardScaler()
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train_scaled = scaler.fit_transform(X_train)
    
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

    # Calculate sample weights based on Credit_History
    weights = X_train['Credit_History'].apply(lambda x: 0.5 if x == 1 else 2.0).values

    # Train models
    models = {
        "LogisticRegression": LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
        "RandomForestClassifier": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(random_state=42, probability=True)
    }

    for model_name, model in models.items():
        if model_name == "SVM" and not isinstance(smote, SMOTE):
            model.fit(X_train_resampled, y_train_resampled, sample_weight=weights)
        else:
            model.fit(X_train_resampled, y_train_resampled)
        # Save the model and scaler
        joblib.dump(model, f"{model_name}.joblib")
        joblib.dump(scaler, f"{model_name}_scaler.joblib")

if __name__ == "__main__":
    train_and_save_models()