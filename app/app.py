import streamlit as st

st.set_page_config(
    page_title="Enterprise Customer Intelligence Platform",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Enterprise Customer Intelligence & Analytics Platform")

st.markdown("---")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Customers", "7,043")

with col2:
    st.metric("Churn Rate", "26.5%")

with col3:
    st.metric("Models", "6")

with col4:
    st.metric("Accuracy", "86%")

st.markdown("---")

st.header("Project Modules")

c1, c2 = st.columns(2)

with c1:
    st.info("📈 Customer Churn Prediction")
    st.info("📊 Customer Analytics Dashboard")
    st.info("🎯 Customer Segmentation")
    st.info("💬 Sentiment Analysis")

with c2:
    st.info("🛍 Recommendation System")
    st.info("📉 Regression Models")
    st.info("🤖 Classification Models")
    st.info("📚 Explainable AI")

st.markdown("---")

st.success("Enterprise AI Platform Ready")