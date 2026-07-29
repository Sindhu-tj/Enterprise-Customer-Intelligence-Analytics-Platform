import os
import sys
import joblib
import pandas as pd
import streamlit as st

# ==========================================================
# Add Project Root
# ==========================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ==========================================================
# Paths
# ==========================================================

MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "regression_model.pkl"
)

ENCODER_PATH = os.path.join(
    MODEL_DIR,
    "regression_encoders.pkl"
)

FEATURE_PATH = os.path.join(
    MODEL_DIR,
    "regression_features.pkl"
)

# ==========================================================
# Load Saved Files
# ==========================================================

model = joblib.load(MODEL_PATH)

encoders = joblib.load(ENCODER_PATH)

feature_names = joblib.load(FEATURE_PATH)

# ==========================================================
# Page Configuration
# ==========================================================

st.set_page_config(
    page_title="Regression",
    page_icon="📉",
    layout="wide"
)

# ==========================================================
# Header
# ==========================================================

st.title("📉 Customer Monthly Charge Prediction")

st.markdown(
"""
Predict the expected **Monthly Charges**
for a customer using the trained
Random Forest Regression model.
"""
)

st.markdown("---")

# ==========================================================
# Customer Information
# ==========================================================

st.subheader("Customer Information")

col1, col2 = st.columns(2)
# ==========================================================
# Customer Inputs
# ==========================================================

with col1:

    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

    senior = st.selectbox(
        "Senior Citizen",
        [0, 1]
    )

    partner = st.selectbox(
        "Partner",
        ["No", "Yes"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["No", "Yes"]
    )

    tenure = st.slider(
        "Tenure (Months)",
        0,
        72,
        12
    )

    phone_service = st.selectbox(
        "Phone Service",
        ["No", "Yes"]
    )

    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["No", "Yes", "No phone service"]
    )

    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    online_security = st.selectbox(
        "Online Security",
        ["No", "Yes", "No internet service"]
    )

    online_backup = st.selectbox(
        "Online Backup",
        ["No", "Yes", "No internet service"]
    )

with col2:

    device_protection = st.selectbox(
        "Device Protection",
        ["No", "Yes", "No internet service"]
    )

    tech_support = st.selectbox(
        "Tech Support",
        ["No", "Yes", "No internet service"]
    )

    streaming_tv = st.selectbox(
        "Streaming TV",
        ["No", "Yes", "No internet service"]
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["No", "Yes", "No internet service"]
    )

    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )

    paperless = st.selectbox(
        "Paperless Billing",
        ["No", "Yes"]
    )

    payment = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

    total_charges = st.number_input(
        "Total Charges ($)",
        min_value=0.0,
        value=850.0,
        step=10.0
    )

st.markdown("---")

predict = st.button(
    "📉 Predict Monthly Charges",
    use_container_width=True
)
# ==========================================================
# Prediction
# ==========================================================

if predict:

    try:

        # Create Input Dictionary
        input_data = {
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "TotalCharges": total_charges
        }

        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])

        # Encode categorical columns
        for column, encoder in encoders.items():

            if column in input_df.columns:

                input_df[column] = encoder.transform(
                    input_df[column]
                )

        # Arrange columns in training order
        input_df = input_df[feature_names]

        # Predict
        prediction = model.predict(input_df)[0]

        st.success(
            f"✅ Predicted Monthly Charges : ${prediction:.2f}"
        )

        st.markdown("---")

        st.subheader("Prediction Summary")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Predicted Charges",
                f"${prediction:.2f}"
            )

        with col2:
            st.metric(
                "Contract",
                contract
            )

        with col3:
            st.metric(
                "Tenure",
                f"{tenure} Months"
            )

        st.markdown("---")

        st.subheader("Customer Information")

        st.dataframe(
            pd.DataFrame([input_data]),
            use_container_width=True
        )

    except Exception as e:

        st.error("Prediction Failed")

        st.exception(e)
        # ==========================================================
# Interpretation
# ==========================================================

        st.markdown("---")
        st.subheader("📊 Prediction Interpretation")

        if prediction < 30:

            st.success("""
### 🟢 Low Monthly Charges

This customer has relatively low monthly charges.

Possible reasons:
- Basic service plan
- Few additional services
- Budget customer
""")

        elif prediction < 70:

            st.info("""
### 🔵 Medium Monthly Charges

This customer falls into the average spending category.

Possible reasons:
- Standard subscription
- Multiple active services
- Moderate customer value
""")

        else:

            st.warning("""
### 🔴 High Monthly Charges

This customer has high monthly charges.

Possible reasons:
- Premium subscription
- Fiber Internet
- Streaming Services
- Tech Support
- High-value customer
""")

# ==========================================================
# Footer
# ==========================================================

st.markdown("---")

st.caption(
    "Enterprise Customer Intelligence & Analytics Platform | "
    "Regression Module | Random Forest Regressor"
)