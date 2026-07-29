import os
import joblib
import numpy as np
import pandas as pd

# ==========================================================
# Paths
# ==========================================================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)

MODEL_DIR = os.path.join(BASE_DIR, "models")

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "customer_churn.csv"
)

# ==========================================================
# Load Saved Files
# ==========================================================

similarity_matrix = joblib.load(
    os.path.join(
        MODEL_DIR,
        "recommendation_model.pkl"
    )
)

feature_names = joblib.load(
    os.path.join(
        MODEL_DIR,
        "recommendation_feature_names.pkl"
    )
)

label_encoders = joblib.load(
    os.path.join(
        MODEL_DIR,
        "recommendation_label_encoders.pkl"
    )
)

# ==========================================================
# Load Dataset
# ==========================================================

df = pd.read_csv(DATA_PATH)

df.columns = df.columns.str.strip()

# Remove customerID if present
if "customerID" in df.columns:
    df.drop(columns=["customerID"], inplace=True)

# Convert TotalCharges
if "TotalCharges" in df.columns:

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    df["TotalCharges"] = df["TotalCharges"].fillna(
        df["TotalCharges"].median()
    )

# Encode categorical columns
for column, encoder in label_encoders.items():

    if column in df.columns:

        df[column] = encoder.transform(df[column])

# ==========================================================
# Recommendation Function
# ==========================================================

def recommend_customer(customer_index, top_n=5):

    """
    Returns the most similar customers.
    """

    if customer_index < 0 or customer_index >= len(df):

        raise ValueError(
            f"Customer index must be between 0 and {len(df)-1}"
        )

    similarity_scores = similarity_matrix[customer_index]

    similar_customers = list(
        enumerate(similarity_scores)
    )

    similar_customers = sorted(
        similar_customers,
        key=lambda x: x[1],
        reverse=True
    )

    # Ignore the selected customer
    similar_customers = similar_customers[1:top_n+1]

    recommendations = []

    for index, score in similar_customers:

        customer = df.iloc[index].copy()

        recommendations.append({

            "Customer Index": int(index),

            "Similarity Score": round(
                float(score),
                4
            ),

            "Tenure":
                int(customer["tenure"])
                if "tenure" in customer
                else None,

            "MonthlyCharges":
                float(customer["MonthlyCharges"])
                if "MonthlyCharges" in customer
                else None,

            "TotalCharges":
                float(customer["TotalCharges"])
                if "TotalCharges" in customer
                else None

        })

    return pd.DataFrame(recommendations)


# ==========================================================
# Test
# ==========================================================

if __name__ == "__main__":

    print("=" * 60)
    print("Recommendation Test")
    print("=" * 60)

    result = recommend_customer(0)

    print(result)