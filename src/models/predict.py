import os
import joblib
import pandas as pd

# =====================================================
# Project Paths
# =====================================================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# =====================================================
# Load Saved Files
# =====================================================
model = joblib.load(os.path.join(MODEL_DIR, "churn_model.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
feature_names = joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl"))
label_encoders = joblib.load(os.path.join(MODEL_DIR, "label_encoders.pkl"))

# =====================================================
# Prediction Function
# =====================================================
def predict_churn(input_data):
    """
    input_data: Dictionary containing customer details
    """

    # Convert dictionary to DataFrame
    input_df = pd.DataFrame([input_data])

    # Encode categorical columns
    for column, encoder in label_encoders.items():
        if column in input_df.columns:
            input_df[column] = encoder.transform(input_df[column])

    # Ensure correct feature order
    input_df = input_df[feature_names]

    # Scale features
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    result = "Churn" if prediction == 1 else "No Churn"

    return {
        "prediction": result,
        "probability": round(probability * 100, 2)
    }