import os
import sys
import streamlit as st
import pandas as pd

# =====================================================
# Add Project Root
# =====================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# =====================================================
# Import Recommendation Function
# =====================================================

from src.recommendation.predict_recommendation import recommend_customer

# =====================================================
# Page Configuration
# =====================================================

st.set_page_config(
    page_title="Recommendation System",
    page_icon="🎁",
    layout="wide"
)

# =====================================================
# Header
# =====================================================

st.title("🎁 Customer Recommendation System")

st.markdown("""
Find customers with similar behaviour using
Cosine Similarity Recommendation.
""")

st.divider()

# =====================================================
# Sidebar Inputs
# =====================================================

st.subheader("Recommendation Settings")

col1, col2 = st.columns(2)

with col1:

    customer_index = st.number_input(
        "Customer Index",
        min_value=0,
        value=0,
        step=1
    )

with col2:

    top_n = st.slider(
        "Number of Recommendations",
        min_value=1,
        max_value=10,
        value=5
    )

st.divider()

# =====================================================
# Recommendation Button
# =====================================================

if st.button(
    "🎁 Get Recommendations",
    use_container_width=True
):

    try:

        recommendations = recommend_customer(
            customer_index,
            top_n
        )

        st.success(
            f"Top {top_n} Similar Customers Found"
        )

        st.dataframe(
            recommendations,
            use_container_width=True
        )

        st.divider()

        st.subheader("Recommendation Summary")

        st.metric(
            "Recommendations",
            len(recommendations)
        )

        avg_similarity = recommendations[
            "Similarity Score"
        ].mean()

        st.metric(
            "Average Similarity",
            f"{avg_similarity:.3f}"
        )

    except Exception as e:

        st.error("Prediction Failed")

        st.exception(e)

# =====================================================
# Footer
# =====================================================

st.divider()

st.caption(
    "Enterprise Customer Intelligence & Analytics Platform"
)