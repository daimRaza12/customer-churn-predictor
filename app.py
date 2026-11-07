import gradio as gr
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

def predict_churn(gender, senior_citizen, partner, dependents, tenure, 
                  phone_service, multiple_lines, internet_service, 
                  online_security, online_backup, device_protection, 
                  tech_support, streaming_tv, streaming_movies, contract, 
                  paperless_billing, payment_method, monthly_charges, total_charges):
    
    # Assemble input record
    input_data = pd.DataFrame([[
        gender, int(senior_citizen == "Yes"), partner, dependents, int(tenure), 
        phone_service, multiple_lines, internet_service, online_security, 
        online_backup, device_protection, tech_support, streaming_tv, 
        streaming_movies, contract, paperless_billing, payment_method, 
        float(monthly_charges), float(total_charges)
    ]], columns=expected_columns)
    
    # Apply same preprocessing pipeline
    processed_data = preprocessor.transform(input_data)
    cleaned_data = imputer.transform(processed_data)
    
    # Get probability and prediction
    prob = model.predict_proba(cleaned_data)[0][1]
    
    # Define Risk Level and Suggestions
    if prob < 0.3:
        risk_level = "🟢 Low Risk"
        suggestion = "Customer is highly satisfied. Maintain regular service notifications and consider introducing them to loyalty programs."
    elif prob < 0.65:
        risk_level = "🟡 Medium Risk"
        suggestion = "Customer is showing early indicators of churn risk. Proactively invite them to optimize their plan or suggest a minor billing upgrade/discount."
    else:
        risk_level = "🔴 High Risk"
        suggestion = "Critical risk! Recommend immediate high-touch outreach. Offer loyalty contracts, month-to-month upgrade incentives, or target direct discount billing options."
        
    return f"Churn Probability: {prob:.1%}", risk_level, suggestion

# Design the interactive interface
interface = gr.Interface(
    fn=predict_churn,
    inputs=[
        gr.Dropdown(["Female", "Male"], label="Gender"),
        gr.Dropdown(["No", "Yes"], label="Senior Citizen"),
        gr.Dropdown(["No", "Yes"], label="Has Partner"),
        gr.Dropdown(["No", "Yes"], label="Has Dependents"),
        gr.Slider(0, 72, step=1, label="Tenure (Months)"),
        gr.Dropdown(["No", "Yes"], label="Phone Service"),
        gr.Dropdown(["No phone service", "No", "Yes"], label="Multiple Lines"),
        gr.Dropdown(["DSL", "Fiber optic", "No"], label="Internet Service"),
        gr.Dropdown(["No internet service", "No", "Yes"], label="Online Security"),
        gr.Dropdown(["No internet service", "No", "Yes"], label="Online Backup"),
        gr.Dropdown(["No internet service", "No", "Yes"], label="Device Protection"),
        gr.Dropdown(["No internet service", "No", "Yes"], label="Tech Support"),
        gr.Dropdown(["No internet service", "No", "Yes"], label="Streaming TV"),
        gr.Dropdown(["No internet service", "No", "Yes"], label="Streaming Movies"),
        gr.Dropdown(["Month-to-month", "One year", "Two year"], label="Contract Type"),
        gr.Dropdown(["No", "Yes"], label="Paperless Billing"),
        gr.Dropdown(["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], label="Payment Method"),
        gr.Number(label="Monthly Charges ($)"),
        gr.Number(label="Total Charges ($)")
    ],
    outputs=[
        gr.Textbox(label="Prediction Score"),
        gr.Textbox(label="Churn Risk Level"),
        gr.Textbox(label="Suggested Retention Action")
    ],
    title="Telco Retention Dashboard & Customer Churn Predictor",
    description="Determine churn probability risk level for any individual subscriber and get actionable business recommendations."
)

if __name__ == '__main__':
    interface.launch(share=True)
