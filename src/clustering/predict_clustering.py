import os
import joblib
import pandas as pd

# ==========================================
# Paths
# ==========================================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# ==========================================
# Load Saved Files
# ==========================================
kmeans = joblib.load(os.path.join(MODEL_DIR, "kmeans_model.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "cluster_scaler.pkl"))
encoders = joblib.load(os.path.join(MODEL_DIR, "cluster_encoders.pkl"))

# ==========================================
# Prediction Function
# ==========================================
def predict_cluster(input_data):

    input_df = pd.DataFrame([input_data])

    # Encode categorical columns
    for col, encoder in encoders.items():
        if col in input_df.columns:
            input_df[col] = encoder.transform(input_df[col])

    # Convert TotalCharges
    if "TotalCharges" in input_df.columns:
        input_df["TotalCharges"] = pd.to_numeric(
            input_df["TotalCharges"],
            errors="coerce"
        )

    # Scale
    input_scaled = scaler.transform(input_df)

    # Predict Cluster
    cluster = kmeans.predict(input_scaled)[0]

    return cluster