import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Analytics Dashboard", layout="wide")

st.title("📊 Customer Analytics Dashboard")

# Load Dataset
df = pd.read_csv("data/raw/customer_churn.csv")

# ===========================
# KPIs
# ===========================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Customers", len(df))

with col2:
    churn_rate = (df["Churn"] == "Yes").mean() * 100
    st.metric("Churn Rate", f"{churn_rate:.2f}%")

with col3:
    st.metric("Average Tenure", f"{df['tenure'].mean():.1f}")

with col4:
    st.metric("Average Monthly Charges", f"${df['MonthlyCharges'].mean():.2f}")

st.divider()

# ===========================
# Churn Distribution
# ===========================

fig = px.pie(
    df,
    names="Churn",
    title="Customer Churn Distribution"
)

st.plotly_chart(fig, use_container_width=True)

# ===========================
# Contract Type
# ===========================

fig = px.histogram(
    df,
    x="Contract",
    color="Churn",
    barmode="group",
    title="Contract Type vs Churn"
)

st.plotly_chart(fig, use_container_width=True)

# ===========================
# Monthly Charges
# ===========================

fig = px.box(
    df,
    x="Churn",
    y="MonthlyCharges",
    color="Churn",
    title="Monthly Charges vs Churn"
)

st.plotly_chart(fig, use_container_width=True)

# ===========================
# Tenure Distribution
# ===========================

fig = px.histogram(
    df,
    x="tenure",
    color="Churn",
    nbins=30,
    title="Tenure Distribution"
)

st.plotly_chart(fig, use_container_width=True)

# ===========================
# Dataset Preview
# ===========================

st.subheader("Dataset Preview")

st.dataframe(df.head())