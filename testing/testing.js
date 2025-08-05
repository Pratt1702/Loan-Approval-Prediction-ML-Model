import fetch from "node-fetch";

const body = {
  Married: 1,
  Dependents: 0,
  Education: 0,
  ApplicantIncome: 4000,
  CoapplicantIncome: 1500,
  LoanAmount: 1200,
  Loan_Amount_Term: 36,
  Credit_History: 1,
  Property_Area: 2,
  use_smote: true,
  model_name: "RandomForestClassifier",
};

fetch("https://loan-approval-prediction-ml-model.onrender.com/predict", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
  },
  body: JSON.stringify(body),
})
  .then(async (res) => {
    const contentType = res.headers.get("content-type");
    const raw = await res.text(); // always read as text first

    if (res.ok) {
      try {
        const data = JSON.parse(raw);
        console.log("✅ Prediction:", data);
      } catch (e) {
        console.error("⚠️ Could not parse JSON:", raw);
      }
    } else {
      console.error("❌ Server error:", res.status, raw);
    }
  })
  .catch((err) => {
    console.error("💥 Network error:", err);
  });
