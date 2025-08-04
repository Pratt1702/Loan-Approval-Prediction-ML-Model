# Loan Prediction Model API

This project is a FastAPI application that predicts loan approval based on user input features. The model is built using machine learning techniques and is designed to be easily integrated with a frontend application.

## Table of Contents

- [Installation](#installation)
- [About the Model](#about-the-model)
- [API Usage](#api-usage)
- [Example Input](#example-input)
- [License](#license)

## Installation

To run this project locally, follow these steps:

1. **Clone the repository**:

   ```bash
   git clone https://github.com/Pratt1702/Loan-Approval-Prediction-ML-Model
   cd "Loan Prediction Model"
   ```

2. **Install the required packages**:
   Make sure you have Python installed. Then, run:

   ```bash
   pip install fastapi uvicorn pandas numpy scikit-learn imblearn
   ```

3. **Run the FastAPI application**:
   Start the server with the following command:

   ```bash
   uvicorn main:app --reload
   ```

4. **Access the API documentation**:
   Open your web browser and go to `http://127.0.0.1:8000/docs` to view the interactive API documentation.

## About the Model

The loan prediction model is built using various machine learning algorithms, including Logistic Regression, Random Forest Classifier, and Support Vector Machine (SVM). The model takes several input features related to the applicant's financial status and personal information to predict whether a loan will be approved.

### Features Used in the Model

- **Married**: 0 for No, 1 for Yes
- **Dependents**: Number of dependents
- **Education**: 0 for Graduate, 1 for Not Graduate
- **ApplicantIncome**: Applicant's income
- **CoapplicantIncome**: Coapplicant's income
- **LoanAmount**: Loan amount in thousands
- **Loan_Amount_Term**: Term of loan in months
- **Credit_History**: 0 for no credit history, 1 for good credit history
- **Property_Area**: 0 for Rural, 1 for Semiurban, 2 for Urban
- **use_smote**: Boolean to indicate whether to use SMOTE for handling imbalanced data
- **model_name**: The machine learning model to use for prediction (LogisticRegression, RandomForestClassifier, SVM)

## API Usage

To make a prediction, send a POST request to the `/predict` endpoint with a JSON body containing the required fields.

### Example Input

Here’s an example of the JSON input you can send to the API:

```json
{
  "Married": 1,
  "Dependents": 0,
  "Education": 0,
  "ApplicantIncome": 5000,
  "CoapplicantIncome": 1500,
  "LoanAmount": 200,
  "Loan_Amount_Term": 360,
  "Credit_History": 1,
  "Property_Area": 2,
  "use_smote": true,
  "model_name": "LogisticRegression"
}
```

### Example Response

The API will return a JSON response indicating whether the loan is approved:

```json
{
  "approved": true
}
```

## License

This project is licensed under the MIT License. See the LICENSE file for more details.
