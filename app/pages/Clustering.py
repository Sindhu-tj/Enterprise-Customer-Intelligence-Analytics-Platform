import os
import sys
import streamlit as st

# ============================================================
# Add Project Root
# ============================================================
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.clustering.predict_clustering import predict_cluster

# ============================================================
# Page Configuration
# ============================================================
st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Customer Segmentation")
st.markdown("### Enterprise Customer Intelligence & Analytics Platform")
st.divider()

st.write(
    "Predict the customer segment using the trained K-Means clustering model."
)

# ============================================================
# Customer Details
# ============================================================

st.subheader("Customer Information")

col1, col2, col3 = st.columns(3)

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

    tenure = st.number_input(
        "Tenure (Months)",
        min_value=0,
        max_value=100,
        value=12
    )

with col2:

    phone = st.selectbox(
        "Phone Service",
        ["No", "Yes"]
    )

    multiple = st.selectbox(
        "Multiple Lines",
        [
            "No",
            "Yes",
            "No phone service"
        ]
    )

    internet = st.selectbox(
        "Internet Service",
        [
            "DSL",
            "Fiber optic",
            "No"
        ]
    )

    security = st.selectbox(
        "Online Security",
        [
            "No",
            "Yes",
            "No internet service"
        ]
    )

    backup = st.selectbox(
        "Online Backup",
        [
            "No",
            "Yes",
            "No internet service"
        ]
    )

with col3:

    protection = st.selectbox(
        "Device Protection",
        [
            "No",
            "Yes",
            "No internet service"
        ]
    )

    support = st.selectbox(
        "Tech Support",
        [
            "No",
            "Yes",
            "No internet service"
        ]
    )

    streaming_tv = st.selectbox(
        "Streaming TV",
        [
            "No",
            "Yes",
            "No internet service"
        ]
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        [
            "No",
            "Yes",
            "No internet service"
        ]
    )

st.divider()

st.subheader("Billing Information")

left, right = st.columns(2)

with left:

    contract = st.selectbox(
        "Contract",
        [
            "Month-to-month",
            "One year",
            "Two year"
        ]
    )

    paperless = st.selectbox(
        "Paperless Billing",
        [
            "No",
            "Yes"
        ]
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

with right:

    monthly = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=70.0
    )

    total = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=850.0
    )

st.divider()

predict = st.button(
    "🔍 Predict Customer Cluster",
    use_container_width=True
)
# ============================================================
# Prediction
# ============================================================

if predict:

    try:

        input_data = {
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multiple,
            "InternetService": internet,
            "OnlineSecurity": security,
            "OnlineBackup": backup,
            "DeviceProtection": protection,
            "TechSupport": support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": total
        }

        cluster = predict_cluster(input_data)

        st.success(f"✅ Predicted Customer Cluster : {cluster}")

        st.divider()

        if cluster == 0:

            st.info("""
### 🟢 Cluster 0 - Loyal Customers

- Long-term customers
- Stable subscriptions
- Moderate spending
- Low churn tendency
            """)

        elif cluster == 1:

            st.info("""
### 🔵 Cluster 1 - Premium Customers

- High monthly spending
- Premium internet plans
- Valuable customers
- Excellent upselling opportunities
            """)

        elif cluster == 2:

            st.info("""
### 🟠 Cluster 2 - Budget Customers

- Lower monthly charges
- Basic services
- Price-sensitive customers
- Suitable for promotional offers
            """)

        elif cluster == 3:

            st.info("""
### 🟣 Cluster 3 - At-Risk Customers

- Higher probability of leaving
- Require retention strategies
- Frequent service issues
- Personalized offers recommended
            """)

        else:

            st.warning(f"Predicted Cluster : {cluster}")

        st.divider()

        st.subheader("Customer Summary")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric("Tenure", f"{tenure} Months")
            st.metric("Monthly Charges", f"${monthly:.2f}")

        with c2:
            st.metric("Total Charges", f"${total:.2f}")
            st.metric("Internet", internet)

        with c3:
            st.metric("Contract", contract)
            st.metric("Cluster", cluster)

        st.divider()

        st.subheader("Prediction Details")

        st.json(input_data)

    except Exception as e:

        st.error("Prediction Failed")

        st.exception(e)

# ============================================================
# Footer
# ============================================================

st.divider()

st.caption("Enterprise Customer Intelligence & Analytics Platform")