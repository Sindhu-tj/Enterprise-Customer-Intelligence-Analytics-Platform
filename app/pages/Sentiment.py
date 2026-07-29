import os
import sys
import streamlit as st

# =====================================================
# Add Project Root
# =====================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# =====================================================
# Import Prediction Function
# =====================================================

from src.nlp.predict_sentiment import predict_sentiment

# =====================================================
# Page Configuration
# =====================================================

st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="💬",
    layout="wide"
)

# =====================================================
# Header
# =====================================================

st.title("💬 AI Sentiment Analysis")

st.markdown("""
Analyze customer reviews, feedback, or comments using
Machine Learning and Natural Language Processing (NLP).
""")

st.divider()

# =====================================================
# User Input
# =====================================================

st.subheader("Enter Customer Review")

review = st.text_area(
    "Customer Review",
    placeholder="Example: The service was excellent and I really enjoyed using this product.",
    height=180
)

analyze = st.button(
    "🔍 Analyze Sentiment",
    use_container_width=True
)

st.divider()
# =====================================================
# Prediction
# =====================================================

if analyze:

    if review.strip() == "":

        st.warning("Please enter a customer review.")

    else:

        try:

            sentiment = predict_sentiment(review)

            st.success("Analysis Completed Successfully!")

            st.markdown("---")

            st.subheader("Prediction Result")

            col1, col2 = st.columns(2)

            with col1:

                if str(sentiment).lower() == "positive":

                    st.success("😊 Positive Sentiment")

                elif str(sentiment).lower() == "negative":

                    st.error("😞 Negative Sentiment")

                elif str(sentiment).lower() == "neutral":

                    st.info("😐 Neutral Sentiment")

                else:

                    st.warning(f"Prediction : {sentiment}")

            with col2:

                st.metric(
                    "Predicted Sentiment",
                    str(sentiment).title()
                )

            st.markdown("---")

            st.subheader("Customer Review")

            st.info(review)

        except Exception as e:

            st.error("Prediction Failed")

            st.exception(e)
            # =====================================================
# AI Interpretation
# =====================================================

            st.markdown("---")

            st.subheader("🤖 AI Interpretation")

            sentiment_lower = str(sentiment).lower()

            if sentiment_lower == "positive":

                st.success("""
### 😊 Positive Feedback

The customer has expressed a positive opinion.

Possible indicators:
- Customer is satisfied
- Positive user experience
- Good service quality
- High customer engagement

Business Recommendation

✔ Continue providing quality service

✔ Offer loyalty rewards

✔ Recommend premium plans
""")

            elif sentiment_lower == "negative":

                st.error("""
### 😞 Negative Feedback

The customer appears dissatisfied.

Possible indicators:
- Poor customer experience
- Service issues
- Product complaints
- High churn risk

Business Recommendation

✔ Contact customer immediately

✔ Assign support ticket

✔ Offer discounts or retention plan
""")

            elif sentiment_lower == "neutral":

                st.info("""
### 😐 Neutral Feedback

The customer has expressed a neutral opinion.

Possible indicators:
- Mixed experience
- No strong emotions
- Requires further engagement

Business Recommendation

✔ Collect more feedback

✔ Improve customer engagement

✔ Recommend suitable services
""")

            else:

                st.warning(
                    f"Prediction Returned : {sentiment}"
                )

# =====================================================
# Business Insights
# =====================================================

st.markdown("---")

st.subheader("📊 Business Insights")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "AI Model",
        "Logistic Regression"
    )

with col2:
    st.metric(
        "NLP Technique",
        "TF-IDF"
    )

with col3:
    st.metric(
        "Prediction",
        "Real-Time"
    )

# =====================================================
# Footer
# =====================================================

st.markdown("---")

st.caption(
    "Enterprise Customer Intelligence & Analytics Platform | "
    "Sentiment Analysis Module | NLP + Machine Learning"
)