# 🔮 Customer Churn Prediction

A complete end-to-end machine learning project that predicts whether a telecom customer will churn, built with Python, Scikit-learn, XGBoost, and deployed as an interactive Streamlit web application.

🚀 **Live Demo:** [customer-churn-prediction-jtt9t4idhsubiwdmtxoedr.streamlit.app](https://customer-churn-prediction-jtt9t4idhsubiwdmtxoedr.streamlit.app/)

---

## 📌 Project Overview

Customer churn is one of the biggest challenges for telecom companies. This project builds a machine learning pipeline to identify customers at risk of leaving, enabling businesses to take proactive retention actions.

**Dataset:** IBM Telco Customer Churn (7,043 customers, 21 features)

---

## 🗂️ Project Structure

```
customer-churn-prediction/
│
├── data/                          ← Dataset and saved charts
│   ├── WA_Fn-UseC_-Telco-Customer-Churn.csv
│   ├── churn_distribution.png
│   ├── tenure_vs_churn.png
│   ├── monthly_charges_vs_churn.png
│   ├── contract_vs_churn.png
│   ├── confusion_matrix.png
│   └── feature_importance.png
│
├── notebooks/
│   ├── 01_EDA_and_Cleaning.ipynb         ← Exploratory data analysis
│   └── 02_Feature_Engineering_and_Model.ipynb ← ML pipeline
│
├── src/
│   ├── churn_model.pkl            ← Saved Random Forest model
│   └── scaler.pkl                 ← Saved StandardScaler
│
├── app.py                         ← Streamlit web application
├── requirements.txt               ← Python dependencies
└── README.md
```

---

## 🛠️ Tools & Technologies

- **Python** — core language
- **Pandas, NumPy** — data manipulation
- **Matplotlib, Seaborn** — data visualization
- **Scikit-learn** — ML models and preprocessing
- **XGBoost** — gradient boosting model
- **imbalanced-learn (SMOTE)** — handling class imbalance
- **Joblib** — model serialization
- **Streamlit** — web app deployment
- **Git & GitHub** — version control

---

## 🧪 ML Pipeline

| Step | Description |
|------|-------------|
| 1 | Exploratory Data Analysis (EDA) |
| 2 | Data Cleaning — TotalCharges conversion, dropped 11 null rows |
| 3 | Feature Engineering — Label Encoding of 15 categorical columns |
| 4 | SMOTE — balanced dataset from 73/27% to 50/50% |
| 5 | Train/Test Split — 80/20 split |
| 6 | StandardScaler — normalized numerical features |
| 7 | Model Training — 3 models compared |
| 8 | Evaluation — accuracy, classification report, confusion matrix |
| 9 | Deployment — Streamlit app |

---

## 📊 Model Results

| Model | Accuracy |
|-------|----------|
| Logistic Regression | 80.64% |
| XGBoost | 83.69% |
| **Random Forest** | **83.88% ← Best** |

**Final Model: Random Forest — 84% Accuracy**

- Precision: 84%
- Recall: 84%
- F1-Score: 84%
- Balanced performance on both churned and non-churned classes ✅

---

## 🔑 Key Insights from EDA

- **Month-to-month contracts** have a 43% churn rate vs 3% for two-year contracts
- **New customers** (0–10 months tenure) churn the most
- **Churned customers** pay ~$15 more per month on average
- **Top 4 churn predictors:** MonthlyCharges, TotalCharges, Contract type, Tenure

---

## 🚀 Run Locally

```bash
# Clone the repo
git clone https://github.com/ansilmohamed/customer-churn-prediction.git
cd customer-churn-prediction

# Create environment
conda create -n churn-env python=3.11
conda activate churn-env

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

## 👤 Author

**Mohamed Ansil CP**
- 🌐 Portfolio: [ansilmohamed.lovable.app](https://ansilmohamed.lovable.app)
- 💼 LinkedIn: [linkedin.com/in/ansilmohamed](https://linkedin.com/in/ansilmohamed)
- 🐙 GitHub: [github.com/ansilmohamed](https://github.com/ansilmohamed)
