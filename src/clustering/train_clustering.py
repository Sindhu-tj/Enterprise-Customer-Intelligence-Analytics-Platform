import os
import joblib
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ==========================================
# Paths
# ==========================================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "customer_churn.csv"
)

MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ==========================================
# Load Dataset
# ==========================================
df = pd.read_csv(DATA_PATH)

df.columns = df.columns.str.strip()

# ==========================================
# Remove Customer ID
# ==========================================
if "customerID" in df.columns:
    df.drop(columns=["customerID"], inplace=True)

# ==========================================
# Convert TotalCharges
# ==========================================
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

df["TotalCharges"] = df["TotalCharges"].fillna(
    df["TotalCharges"].median()
)

# ==========================================
# Remove Target Column
# ==========================================
if "Churn" in df.columns:
    df.drop(columns=["Churn"], inplace=True)

# ==========================================
# Encode Categorical Columns
# ==========================================
encoders = {}

for col in df.select_dtypes(include="object").columns:
    encoder = LabelEncoder()
    df[col] = encoder.fit_transform(df[col])
    encoders[col] = encoder

# ==========================================
# Feature Scaling
# ==========================================
scaler = StandardScaler()

scaled_data = scaler.fit_transform(df)

# ==========================================
# Train KMeans
# ==========================================
kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

kmeans.fit(scaled_data)

# ==========================================
# Save Model
# ==========================================
joblib.dump(kmeans, os.path.join(MODEL_DIR, "kmeans_model.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "cluster_scaler.pkl"))
joblib.dump(encoders, os.path.join(MODEL_DIR, "cluster_encoders.pkl"))

print("=" * 50)
print("KMeans Model Saved Successfully!")
print("=" * 50)