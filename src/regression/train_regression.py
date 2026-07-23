import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# =====================================================
# Project Paths
# =====================================================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "customer_churn.csv"
)

MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# =====================================================
# Load Dataset
# =====================================================

df = pd.read_csv(DATA_PATH)

df.columns = df.columns.str.strip()

# =====================================================
# Clean Dataset
# =====================================================

if "customerID" in df.columns:
    df.drop(columns=["customerID"], inplace=True)

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

df["TotalCharges"] = df["TotalCharges"].fillna(
    df["TotalCharges"].median()
)

# =====================================================
# Encode Categorical Columns
# =====================================================

encoders = {}

for column in df.select_dtypes(include="object").columns:

    encoder = LabelEncoder()

    df[column] = encoder.fit_transform(df[column])

    encoders[column] = encoder

# =====================================================
# Regression Target
# Predict Monthly Charges
# =====================================================

TARGET = "MonthlyCharges"

# Remove the target and Churn from the input features
X = df.drop(columns=[
    "MonthlyCharges",
    "Churn"
])

y = df["MonthlyCharges"]
# =====================================================
# Split Dataset
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =====================================================
# Train Model
# =====================================================

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

# =====================================================
# Evaluation
# =====================================================

prediction = model.predict(X_test)

mae = mean_absolute_error(y_test, prediction)
r2 = r2_score(y_test, prediction)

print("=" * 60)
print("Regression Model Trained Successfully")
print("=" * 60)
print(f"MAE : {mae:.2f}")
print(f"R² Score : {r2:.4f}")

# =====================================================
# Save Files
# =====================================================

joblib.dump(
    model,
    os.path.join(MODEL_DIR, "regression_model.pkl")
)

joblib.dump(
    encoders,
    os.path.join(MODEL_DIR, "regression_encoders.pkl")
)

joblib.dump(
    list(X.columns),
    os.path.join(MODEL_DIR, "regression_features.pkl")
)

print("=" * 60)
print("Regression Model Saved Successfully!")
print("=" * 60)