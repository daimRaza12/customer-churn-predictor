import streamlit as gr
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load pre-trained assets
with open('best_rf_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)
with open('imputer.pkl', 'rb') as f:
    imputer = pickle.load(f)

# Retrieve column layout expected by training dataset structure
expected_columns = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
    'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
    'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
    'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
    'MonthlyCharges', 'TotalCharges'
]

st.set_page_config(page_title="Telco Churn Predictor", page_icon="📊")
st.title("Telco Retention Dashboard & Customer Churn Predictor")
st.write("Determine churn probability risk level for any individual subscriber and get actionable business recommendations.")

# Design the side-by-side inputs using columns
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Has Partner", ["No", "Yes"])
    dependents = st.selectbox("Has Dependents", ["No", "Yes"])
    tenure = st.slider("Tenure (Months)", 0, 72, 12)
    phone_service = st.selectbox("Phone Service", ["No", "Yes"])
    multiple_lines = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["No internet service", "No", "Yes"])
    online_backup = st.selectbox("Online Backup", ["No internet service", "No", "Yes"])

with col2:
    device_protection = st.selectbox("Device Protection", ["No internet service", "No", "Yes"])
    tech_support = st.selectbox("Tech Support", ["No internet service", "No", "Yes"])
    streaming_tv = st.selectbox("Streaming TV", ["No internet service", "No", "Yes"])
    streaming_movies = st.selectbox("Streaming Movies", ["No internet service", "No", "Yes"])
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
    payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    monthly_charges = st.number_input("Monthly Charges ($)", value=50.0)
    total_charges = st.number_input("Total Charges ($)", value=500.0)

if st.button("Predict Risk and Actions"):
    input_data = pd.DataFrame([[
        gender, int(senior_citizen == "Yes"), partner, dependents, int(tenure),
        phone_service, multiple_lines, internet_service, online_security,
        online_backup, device_protection, tech_support, streaming_tv,
        streaming_movies, contract, paperless_billing, payment_method,
        float(monthly_charges), float(total_charges)
    ]], columns=expected_columns)
    
    # Process & Predict
    processed_data = preprocessor.transform(input_data)
    cleaned_data = imputer.transform(processed_data)
    prob = model.predict_proba(cleaned_data)[0][1]
    
    # Display Output UI Panels
    st.subheader("Prediction Summary")
    st.metric(label="Churn Probability Score", value=f"{prob:.1%}")
    
    if prob < 0.3:
        st.success("🟢 Low Risk Level")
        st.info("**Suggested Action:** Customer is highly satisfied. Maintain regular service notifications and consider introducing them to loyalty programs.")
    elif prob < 0.65:
        st.warning("🟡 Medium Risk Level")
        st.info("**Suggested Action:** Customer is showing early indicators of churn risk. Proactively invite them to optimize their plan or suggest a minor billing upgrade/discount.")
    else:
        st.error("🔴 High Risk Level")
        st.info("**Suggested Action:** Critical risk! Recommend immediate high-touch outreach. Offer loyalty contracts, month-to-month upgrade incentives, or target direct discount billing options.")
