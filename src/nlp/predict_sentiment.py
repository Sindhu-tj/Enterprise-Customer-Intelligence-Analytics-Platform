import os
import joblib

# ======================================================
# Project Paths
# ======================================================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)

MODEL_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "sentiment_model.pkl"
)

VECTORIZER_PATH = os.path.join(
    MODEL_DIR,
    "sentiment_vectorizer.pkl"
)

# ======================================================
# Load Model & Vectorizer
# ======================================================

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

# ======================================================
# Prediction Function
# ======================================================

def predict_sentiment(text):

    if text is None or str(text).strip() == "":
        return "Please enter some text."

    # Convert text into TF-IDF features
    text_vector = vectorizer.transform([text])

    # Predict sentiment
    prediction = model.predict(text_vector)[0]

    return prediction