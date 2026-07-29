import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ==========================================================
# PAGE CONFIGURATION
# ==========================================================
st.set_page_config(
    page_title="Enterprise Customer Intelligence & Analytics Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# CUSTOM CSS
# ==========================================================
st.markdown("""
<style>

.block-container{
    padding-top:2rem;
    padding-bottom:2rem;
}

.hero{
    background:linear-gradient(135deg,#0F172A,#1E3A8A);
    border-radius:20px;
    padding:40px;
    color:white;
    margin-bottom:25px;
    box-shadow:0px 8px 25px rgba(0,0,0,.35);
}

.metric-card{
    background:#111827;
    border-radius:18px;
    padding:20px;
    text-align:center;
    border:1px solid #334155;
    transition:0.3s;
}

.metric-card:hover{
    transform:translateY(-6px);
    border-color:#3B82F6;
}

.metric-title{
    color:#9CA3AF;
    font-size:15px;
}

.metric-value{
    color:white;
    font-size:34px;
    font-weight:bold;
}

.metric-icon{
    font-size:42px;
}

.section-title{
    font-size:30px;
    font-weight:bold;
    color:white;
    margin-top:30px;
}

</style>
""", unsafe_allow_html=True)

# ==========================================================
# HERO SECTION
# ==========================================================

st.markdown("""
<div class="hero">

<h1>📊 Enterprise Customer Intelligence & Analytics Platform</h1>

<h4>
AI Powered Customer Analytics Platform for Business Intelligence
</h4>

<p style="font-size:18px">

Analyze customer behaviour using multiple Machine Learning models.

✔ Customer Churn Prediction

✔ Customer Segmentation

✔ Sales Forecasting

✔ Product Recommendation

✔ Sentiment Analysis

✔ Business Analytics Dashboard

</p>

</div>
""", unsafe_allow_html=True)

# ==========================================================
# QUICK STATS
# ==========================================================

st.markdown(
'<div class="section-title">📈 Platform Statistics</div>',
unsafe_allow_html=True
)

c1,c2,c3,c4,c5 = st.columns(5)

with c1:

    st.markdown("""
    <div class="metric-card">

    <div class="metric-icon">📂</div>

    <div class="metric-title">
    Modules
    </div>

    <div class="metric-value">
    6
    </div>

    </div>
    """,unsafe_allow_html=True)

with c2:

    st.markdown("""
    <div class="metric-card">

    <div class="metric-icon">🤖</div>

    <div class="metric-title">
    ML Models
    </div>

    <div class="metric-value">
    5
    </div>

    </div>
    """,unsafe_allow_html=True)

with c3:

    st.markdown("""
    <div class="metric-card">

    <div class="metric-icon">📊</div>

    <div class="metric-title">
    Dataset Size
    </div>

    <div class="metric-value">
    7043
    </div>

    </div>
    """,unsafe_allow_html=True)

with c4:

    st.markdown("""
    <div class="metric-card">

    <div class="metric-icon">⚡</div>

    <div class="metric-title">
    Accuracy
    </div>

    <div class="metric-value">
    92%
    </div>

    </div>
    """,unsafe_allow_html=True)

with c5:

    st.markdown("""
    <div class="metric-card">

    <div class="metric-icon">🚀</div>

    <div class="metric-title">
    Status
    </div>

    <div class="metric-value">
    Beta
    </div>

    </div>
    """,unsafe_allow_html=True)

st.markdown("---")
# ==========================================================
# PLATFORM ANALYTICS
# ==========================================================

st.markdown(
    '<div class="section-title">📊 Platform Analytics</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

# -------------------------------
# Pie Chart
# -------------------------------
with col1:

    module_data = pd.DataFrame({
        "Module": [
            "Classification",
            "Clustering",
            "Regression",
            "Recommendation",
            "Sentiment"
        ],
        "Status": [
            90,
            80,
            50,
            40,
            40
        ]
    })

    fig = px.pie(
        module_data,
        values="Status",
        names="Module",
        hole=0.55,
        title="Project Completion"
    )

    fig.update_layout(height=450)

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# -------------------------------
# Bar Chart
# -------------------------------
with col2:

    customers = pd.DataFrame({

        "Category":[
            "Loyal",
            "Premium",
            "Average",
            "Budget",
            "High Value"
        ],

        "Customers":[
            1850,
            950,
            1450,
            1200,
            1593
        ]

    })

    fig = px.bar(

        customers,

        x="Category",

        y="Customers",

        title="Customer Distribution"

    )

    fig.update_layout(height=450)

    st.plotly_chart(

        fig,

        use_container_width=True

    )

st.markdown("---")

# ==========================================================
# PROJECT PROGRESS
# ==========================================================

st.markdown(

'<div class="section-title">🚀 Project Progress</div>',

unsafe_allow_html=True

)

st.write("Overall Project Completion")

st.progress(60)

st.caption("60% Completed")

st.markdown("### Module Progress")

st.write("Classification")
st.progress(90)

st.write("Clustering")
st.progress(80)

st.write("Regression")
st.progress(50)

st.write("Recommendation")
st.progress(40)

st.write("Sentiment Analysis")
st.progress(40)

st.markdown("---")

# ==========================================================
# FEATURE CARDS
# ==========================================================

st.markdown(

'<div class="section-title">✨ Platform Modules</div>',

unsafe_allow_html=True

)

c1, c2, c3 = st.columns(3)

with c1:

    st.success("""

### 🎯 Customer Classification

Predict customer churn using

Machine Learning.

✔ Random Forest

✔ Probability Prediction

✔ Risk Detection

""")

    st.info("""

### 📈 Customer Clustering

Segment customers into

similar groups.

✔ K-Means

✔ Customer Groups

✔ Business Insights

""")

with c2:

    st.warning("""

### 📉 Regression

Predict future sales

and revenue.

✔ Forecasting

✔ Trend Analysis

✔ Business Planning

""")

    st.success("""

### 💡 Recommendation

Generate personalized

product recommendations.

✔ Similar Products

✔ User Preferences

✔ AI Suggestions

""")

with c3:

    st.info("""

### 💬 Sentiment Analysis

Analyze customer reviews.

✔ Positive

✔ Neutral

✔ Negative

""")

    st.warning("""

### 📊 Analytics Dashboard

Business Intelligence

visualizations.

✔ KPI Dashboard

✔ Charts

✔ Reports

""")

st.markdown("---")
# ==========================================================
# TECHNOLOGY STACK
# ==========================================================

st.markdown(
    '<div class="section-title">🛠 Technology Stack</div>',
    unsafe_allow_html=True
)

t1, t2, t3, t4 = st.columns(4)

with t1:
    st.info("""
### 💻 Frontend

- Streamlit
- HTML5
- CSS3
- Plotly
""")

with t2:
    st.success("""
### 🧠 Machine Learning

- Scikit-Learn
- Pandas
- NumPy
- Joblib
""")

with t3:
    st.warning("""
### ⚙ Backend

- Python
- Pickle
- OS
- Sys
""")

with t4:
    st.info("""
### 📦 Development

- VS Code
- Git
- GitHub
- Jupyter
""")

st.markdown("---")

# ==========================================================
# PROJECT WORKFLOW
# ==========================================================

st.markdown(
    '<div class="section-title">🔄 Project Workflow</div>',
    unsafe_allow_html=True
)

workflow = """"""

st.code(workflow)

st.markdown("---")

# ==========================================================
# MACHINE LEARNING MODELS
# ==========================================================

st.markdown(
    '<div class="section-title">🤖 Machine Learning Models</div>',
    unsafe_allow_html=True
)

model_df = pd.DataFrame({

    "Module":[
        "Classification",
        "Clustering",
        "Regression",
        "Recommendation",
        "Sentiment Analysis"
    ],

    "Algorithm":[
        "Random Forest",
        "K-Means",
        "Linear Regression",
        "Collaborative Filtering",
        "Naive Bayes / NLP"
    ],

    "Status":[
        "Completed",
        "Completed",
        "In Progress",
        "In Progress",
        "In Progress"
    ]

})

st.dataframe(
    model_df,
    use_container_width=True,
    hide_index=True
)

st.markdown("---")

# ==========================================================
# DATASET INFORMATION
# ==========================================================

st.markdown(
    '<div class="section-title">📂 Dataset Information</div>',
    unsafe_allow_html=True
)

d1, d2 = st.columns([2,1])

with d1:

    st.table({

        "Property":[
            "Dataset Name",
            "Total Customers",
            "Features",
            "Target Variable",
            "Missing Values",
            "Source"
        ],

        "Value":[
            "Customer Churn Dataset",
            "7043",
            "20",
            "Churn",
            "Handled",
            "IBM Telco Customer Churn"
        ]

    })

with d2:

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=92,
            title={'text':"Model Accuracy"},
            gauge={
                'axis':{'range':[0,100]},
                'bar':{'color':"green"}
            }
        )
    )

    fig.update_layout(height=350)

    st.plotly_chart(
        fig,
        use_container_width=True
    )

st.markdown("---")

# ==========================================================
# PROJECT SUMMARY
# ==========================================================

st.markdown(
    '<div class="section-title">📋 Project Summary</div>',
    unsafe_allow_html=True
)

st.success("""
### Enterprise Customer Intelligence & Analytics Platform

This project combines multiple Artificial Intelligence and Machine Learning
techniques into one unified business analytics platform.

✔ Customer Churn Prediction

✔ Customer Segmentation

✔ Sales Forecasting

✔ Recommendation System

✔ Sentiment Analysis

✔ Interactive Analytics Dashboard

Designed using Python, Streamlit and Scikit-Learn.
""")

st.markdown("---")
# ==========================================================
# RECENT ACTIVITY
# ==========================================================

st.markdown(
    '<div class="section-title">📅 Recent Development Progress</div>',
    unsafe_allow_html=True
)

timeline = [
    ("✅ Classification Module", "Completed"),
    ("✅ Clustering Module", "Completed"),
    ("🚧 Regression Module", "In Progress"),
    ("🚧 Recommendation System", "In Progress"),
    ("🚧 Sentiment Analysis", "In Progress"),
    ("⏳ Analytics Dashboard", "Pending")
]

for title, status in timeline:
    col1, col2 = st.columns([4, 1])
    with col1:
        st.write(title)
    with col2:
        st.write(status)

st.divider()

# ==========================================================
# QUICK ACTIONS
# ==========================================================

st.markdown(
    '<div class="section-title">⚡ Quick Actions</div>',
    unsafe_allow_html=True
)

c1, c2, c3 = st.columns(3)

with c1:
    st.info("""
### 📊 Analytics

View business KPIs

Customer Insights

Performance Reports
""")

with c2:
    st.success("""
### 🤖 Machine Learning

Classification

Clustering

Regression
""")

with c3:
    st.warning("""
### 💬 AI Features

Recommendation

Sentiment Analysis

Business Intelligence
""")

st.divider()

# ==========================================================
# PROJECT STATUS
# ==========================================================

st.markdown(
    '<div class="section-title">📈 Overall Project Status</div>',
    unsafe_allow_html=True
)

progress = 60

st.progress(progress / 100)

st.success(f"Project Completion : {progress}%")

st.write("""
Current Development Stage

✔ Project Structure Completed

✔ Classification Completed

✔ Clustering Completed

✔ UI Dashboard Developed

🚧 Remaining Modules Under Development
""")

st.divider()

# ==========================================================
# ABOUT THE PROJECT
# ==========================================================

st.markdown(
    '<div class="section-title">ℹ About</div>',
    unsafe_allow_html=True
)

st.info("""
Enterprise Customer Intelligence & Analytics Platform is an integrated
Machine Learning application developed using Streamlit.

The application combines multiple AI modules into a single dashboard
to help businesses analyze customer behavior and support decision-making.

Main Features

• Customer Churn Prediction

• Customer Segmentation

• Sales Prediction

• Product Recommendation

• Sentiment Analysis

• Interactive Analytics Dashboard
""")

st.divider()

# ==========================================================
# FOOTER
# ==========================================================

st.markdown(
"""
---
<center>

## 🎉 Enterprise Customer Intelligence & Analytics Platform

Machine Learning • Business Intelligence • Artificial Intelligence

Built with ❤️ using Python, Streamlit, Scikit-Learn, Pandas and Plotly.

Version 1.0

© 2026 All Rights Reserved.

</center>
""",
unsafe_allow_html=True
)