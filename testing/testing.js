import fetch from "node-fetch";
//TODO: Enter your URL before testing
fetch("<URL>", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
  },
  body: JSON.stringify({
    Married: 0,
    Dependents: 5,
    Education: 0,
    ApplicantIncome: 2000,
    CoapplicantIncome: 0,
    LoanAmount: 50,
    Loan_Amount_Term: 1,
    Credit_History: 1,
    Property_Area: 2,
    use_smote: false,
    model_name: "RandomForestClassifier",
  }),
})
  .then((res) => res.json())
  .then((data) => console.log("Prediction:", data))
  .catch((err) => console.error("Error:", err));
