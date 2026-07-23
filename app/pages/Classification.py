import streamlit as st
import pandas as pd
import joblib

# ==========================================
# Load Saved Model & Files
# ==========================================
model = joblib.load("models/churn_model.pkl")
scaler = joblib.load("models/scaler.pkl")
feature_names = joblib.load("models/feature_names.pkl")
label_encoders = joblib.load("models/label_encoders.pkl")

# ==========================================
# Page Configuration
# ==========================================
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)

st.title("📊 Enterprise Customer Churn Prediction")

st.write(
    "Enter the customer details below to predict whether the customer will churn."
)

st.divider()

# ==========================================
# Customer Information
# ==========================================

gender = st.selectbox(
    "Gender",
    ["Female", "Male"]
)

SeniorCitizen = st.selectbox(
    "Senior Citizen",
    [0, 1]
)

Partner = st.selectbox(
    "Partner",
    ["No", "Yes"]
)

Dependents = st.selectbox(
    "Dependents",
    ["No", "Yes"]
)

tenure = st.slider(
    "Tenure (Months)",
    min_value=0,
    max_value=72,
    value=12
)

PhoneService = st.selectbox(
    "Phone Service",
    ["No", "Yes"]
)

MultipleLines = st.selectbox(
    "Multiple Lines",
    [
        "No",
        "Yes",
        "No phone service"
    ]
)

InternetService = st.selectbox(
    "Internet Service",
    [
        "DSL",
        "Fiber optic",
        "No"
    ]
)

OnlineSecurity = st.selectbox(
    "Online Security",
    [
        "No",
        "Yes",
        "No internet service"
    ]
)

OnlineBackup = st.selectbox(
    "Online Backup",
    [
        "No",
        "Yes",
        "No internet service"
    ]
)

DeviceProtection = st.selectbox(
    "Device Protection",
    [
        "No",
        "Yes",
        "No internet service"
    ]
)

TechSupport = st.selectbox(
    "Tech Support",
    [
        "No",
        "Yes",
        "No internet service"
    ]
)

StreamingTV = st.selectbox(
    "Streaming TV",
    [
        "No",
        "Yes",
        "No internet service"
    ]
)

StreamingMovies = st.selectbox(
    "Streaming Movies",
    [
        "No",
        "Yes",
        "No internet service"
    ]
)

Contract = st.selectbox(
    "Contract",
    [
        "Month-to-month",
        "One year",
        "Two year"
    ]
)

PaperlessBilling = st.selectbox(
    "Paperless Billing",
    [
        "No",
        "Yes"
    ]
)

PaymentMethod = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

MonthlyCharges = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    value=70.0,
    step=0.1
)

TotalCharges = st.number_input(
    "Total Charges",
    min_value=0.0,
    value=900.0,
    step=0.1
)
# ==========================================
# Prediction Section
# ==========================================

st.divider()

if st.button("Predict Churn"):

    # Create input data dictionary
    data = {
        "gender": gender,
        "SeniorCitizen": SeniorCitizen,
        "Partner": Partner,
        "Dependents": Dependents,
        "tenure": tenure,
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges
    }

    # Convert to DataFrame
    input_df = pd.DataFrame([data])

    # ==========================================
    # Apply Saved Label Encoders
    # ==========================================
    for column, encoder in label_encoders.items():
        if column in input_df.columns:
            try:
                input_df[column] = encoder.transform(input_df[column])
            except ValueError:
                st.error(f"Unknown value found in column: {column}")
                st.stop()

    # ==========================================
    # Match Training Feature Order
    # ==========================================
    input_df = input_df[feature_names]

    # ==========================================
    # Scale Features
    # ==========================================
    input_scaled = scaler.transform(input_df)

    # ==========================================
    # Predict
    # ==========================================
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]

    stay_prob = probability[0] * 100
    churn_prob = probability[1] * 100

    st.divider()

    # ==========================================
    # Display Result
    # ==========================================
    if prediction == 1:
        st.error("⚠️ Customer Will Churn")
        st.metric("Churn Probability", f"{churn_prob:.2f}%")
    else:
        st.success("✅ Customer Will Stay")
        st.metric("Stay Probability", f"{stay_prob:.2f}%")

    st.subheader("Prediction Details")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Stay", f"{stay_prob:.2f}%")

    with col2:
        st.metric("Churn", f"{churn_prob:.2f}%")

    # ==========================================
    # Risk Level
    # ==========================================
    if churn_prob >= 70:
        st.error("🔴 High Risk Customer")
    elif churn_prob >= 40:
        st.warning("🟡 Medium Risk Customer")
    else:
        st.success("🟢 Low Risk Customer")