import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and scaler
model = joblib.load('src/churn_model.pkl')
scaler = joblib.load('src/scaler.pkl')

# App title
st.title("🔮 Customer Churn Predictor")
st.write("Fill in the customer details below to predict if they will churn.")

# Input fields
st.sidebar.header("Customer Details")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
senior = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])
tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
phone_service = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
multiple_lines = st.sidebar.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
internet_service = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.sidebar.selectbox("Online Security", ["Yes", "No", "No internet service"])
online_backup = st.sidebar.selectbox("Online Backup", ["Yes", "No", "No internet service"])
device_protection = st.sidebar.selectbox("Device Protection", ["Yes", "No", "No internet service"])
tech_support = st.sidebar.selectbox("Tech Support", ["Yes", "No", "No internet service"])
streaming_tv = st.sidebar.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
streaming_movies = st.sidebar.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
payment_method = st.sidebar.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
monthly_charges = st.sidebar.slider("Monthly Charges ($)", 18.0, 120.0, 65.0)
total_charges = st.sidebar.slider("Total Charges ($)", 18.0, 8700.0, 2000.0)

# Encode inputs
def encode(val, options):
    return options.index(val)

input_data = pd.DataFrame({
    'gender': [encode(gender, ["Female", "Male"])],
    'SeniorCitizen': [1 if senior == "Yes" else 0],
    'Partner': [encode(partner, ["No", "Yes"])],
    'Dependents': [encode(dependents, ["No", "Yes"])],
    'tenure': [tenure],
    'PhoneService': [encode(phone_service, ["No", "Yes"])],
    'MultipleLines': [encode(multiple_lines, ["No", "No phone service", "Yes"])],
    'InternetService': [encode(internet_service, ["DSL", "Fiber optic", "No"])],
    'OnlineSecurity': [encode(online_security, ["No", "No internet service", "Yes"])],
    'OnlineBackup': [encode(online_backup, ["No", "No internet service", "Yes"])],
    'DeviceProtection': [encode(device_protection, ["No", "No internet service", "Yes"])],
    'TechSupport': [encode(tech_support, ["No", "No internet service", "Yes"])],
    'StreamingTV': [encode(streaming_tv, ["No", "No internet service", "Yes"])],
    'StreamingMovies': [encode(streaming_movies, ["No", "No internet service", "Yes"])],
    'Contract': [encode(contract, ["Month-to-month", "One year", "Two year"])],
    'PaperlessBilling': [encode(paperless_billing, ["No", "Yes"])],
    'PaymentMethod': [encode(payment_method, ["Bank transfer (automatic)", "Credit card (automatic)", "Electronic check", "Mailed check"])],
    'MonthlyCharges': [monthly_charges],
    'TotalCharges': [total_charges],
})

# Scale and predict
input_scaled = scaler.transform(input_data)

# Predict button
if st.button("Predict Churn"):
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.error(f"⚠️ This customer is likely to CHURN! (Probability: {probability:.1%})")
        st.write("**Recommendation:** Consider offering a discount or long-term contract.")
    else:
        st.success(f"✅ This customer is likely to STAY! (Probability of churn: {probability:.1%})")
        st.write("**Recommendation:** Keep up the good service!")

    # Show input summary
    st.subheader("Input Summary")
    st.dataframe(input_data)