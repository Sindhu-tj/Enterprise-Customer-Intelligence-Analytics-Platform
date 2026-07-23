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

# ======================================================
# Load Model
# ======================================================

model = joblib.load(MODEL_PATH)

# ======================================================
# Prediction Function
# ======================================================

def predict_sentiment(text):

    if text is None or str(text).strip() == "":
        return "Please enter some text."

    prediction = model.predict([text])[0]

    return prediction