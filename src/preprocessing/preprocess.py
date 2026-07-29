import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ==========================
# Load Dataset
# ==========================
df = pd.read_csv("data/raw/customer_churn.csv")

# Remove leading/trailing spaces from column names
df.columns = df.columns.str.strip()

# Print column names
print("\nColumns in Dataset:")
print(df.columns.tolist())

# ==========================
# Drop customerID if it exists
# ==========================
if "customerID" in df.columns:
    df.drop("customerID", axis=1, inplace=True)
    print("\ncustomerID column removed.")
else:
    print("\ncustomerID column NOT found.")

# ==========================
# Convert TotalCharges to numeric
# ==========================
if "TotalCharges" in df.columns:
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

# ==========================
# Encode categorical columns
# ==========================
label_encoder = LabelEncoder()

for col in df.select_dtypes(include="object").columns:
    df[col] = label_encoder.fit_transform(df[col])

# ==========================
# Split Features and Target
# ==========================
if "Churn" not in df.columns:
    print("\nERROR: 'Churn' column not found.")
    print("Available columns:")
    print(df.columns.tolist())
    exit()

X = df.drop("Churn", axis=1)
y = df["Churn"]

# ==========================
# Train-Test Split
# ==========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==========================
# Feature Scaling
# ==========================
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==========================
# Output
# ==========================
print("\n===================================")
print("Preprocessing Completed Successfully!")
print("===================================")
print("Training Shape :", X_train.shape)
print("Testing Shape  :", X_test.shape)