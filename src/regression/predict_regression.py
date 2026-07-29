import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ==========================================================
# Paths
# ==========================================================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "customer_churn.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

os.makedirs(MODEL_DIR, exist_ok=True)

# ==========================================================
# Load Dataset
# ==========================================================

print("=" * 60)
print("Loading Dataset...")
print("=" * 60)

df = pd.read_csv(DATA_PATH)

df.columns = df.columns.str.strip()

# ==========================================================
# Remove Customer ID
# ==========================================================

if "customerID" in df.columns:
    df.drop(columns=["customerID"], inplace=True)

# ==========================================================
# Convert TotalCharges
# ==========================================================

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

df["TotalCharges"] = df["TotalCharges"].fillna(
    df["TotalCharges"].median()
)

# ==========================================================
# Encode Categorical Columns
# ==========================================================

label_encoders = {}

for column in df.select_dtypes(include="object").columns:

    encoder = LabelEncoder()

    df[column] = encoder.fit_transform(df[column])

    label_encoders[column] = encoder

# ==========================================================
# Features & Target
# ==========================================================

X = df.drop(columns=["MonthlyCharges"])

y = df["MonthlyCharges"]

feature_names = list(X.columns)

# ==========================================================
# Feature Scaling
# ==========================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# ==========================================================
# Train-Test Split
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.20,
    random_state=42
)

# ==========================================================
# Train Model
# ==========================================================

print("=" * 60)
print("Training Linear Regression Model...")
print("=" * 60)

model = LinearRegression()

model.fit(X_train, y_train)

# ==========================================================
# Evaluation
# ==========================================================

predictions = model.predict(X_test)

mse = mean_squared_error(y_test, predictions)

r2 = r2_score(y_test, predictions)

print(f"Mean Squared Error : {mse:.4f}")
print(f"R2 Score           : {r2:.4f}")

# ==========================================================
# Save Files
# ==========================================================

joblib.dump(
    model,
    os.path.join(
        MODEL_DIR,
        "regression_model.pkl"
    )
)

joblib.dump(
    scaler,
    os.path.join(
        MODEL_DIR,
        "regression_scaler.pkl"
    )
)

joblib.dump(
    label_encoders,
    os.path.join(
        MODEL_DIR,
        "regression_label_encoders.pkl"
    )
)

joblib.dump(
    feature_names,
    os.path.join(
        MODEL_DIR,
        "regression_feature_names.pkl"
    )
)

print("=" * 60)
print("Regression Model Saved Successfully!")
print("=" * 60)

print("Saved Files:")
print("✓ regression_model.pkl")
print("✓ regression_scaler.pkl")
print("✓ regression_label_encoders.pkl")
print("✓ regression_feature_names.pkl")

print("=" * 60)
print("Training Completed Successfully")
print("=" * 60)