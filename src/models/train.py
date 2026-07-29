import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# =====================================================
# Project Paths
# =====================================================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "customer_churn.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# =====================================================
# Load Dataset
# =====================================================
df = pd.read_csv(DATA_PATH)

# Clean column names
df.columns = df.columns.str.strip()

print("Columns:", df.columns.tolist())

# =====================================================
# Remove customerID
# =====================================================
if "customerID" in df.columns:
    df.drop(columns=["customerID"], inplace=True)

# =====================================================
# Convert TotalCharges
# =====================================================
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

# =====================================================
# Encode categorical columns
# =====================================================
encoders = {}

for column in df.select_dtypes(include="object").columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    encoders[column] = le

# =====================================================
# Features & Target
# =====================================================
X = df.drop(columns=["Churn"])
y = df["Churn"]

feature_names = X.columns.tolist()

# =====================================================
# Split
# =====================================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# =====================================================
# Scaling
# =====================================================
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =====================================================
# Train Model
# =====================================================
model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model.fit(X_train, y_train)

# =====================================================
# Evaluate
# =====================================================
y_pred = model.predict(X_test)

print("=" * 50)
print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print("=" * 50)
print(classification_report(y_test, y_pred))

# =====================================================
# Save Models
# =====================================================
os.makedirs(MODEL_DIR, exist_ok=True)

joblib.dump(model, os.path.join(MODEL_DIR, "churn_model.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
joblib.dump(feature_names, os.path.join(MODEL_DIR, "feature_names.pkl"))
joblib.dump(encoders, os.path.join(MODEL_DIR, "label_encoders.pkl"))

print("=" * 50)
print("Model Saved Successfully!")
print("Scaler Saved Successfully!")
print("Feature Names Saved Successfully!")
print("Label Encoders Saved Successfully!")
print("=" * 50)