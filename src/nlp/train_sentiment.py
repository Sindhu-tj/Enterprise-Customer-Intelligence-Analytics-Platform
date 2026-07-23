import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ==========================================================
# Project Paths
# ==========================================================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "sentiment.csv"
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
print("Loading Sentiment Dataset...")
print("=" * 60)

df = pd.read_csv(DATA_PATH)

# Remove extra spaces from column names
df.columns = df.columns.str.strip()

# Rename IMDb column to match the project
if "review" in df.columns:
    df.rename(columns={"review": "text"}, inplace=True)

# ==========================================================
# Check Required Columns
# ==========================================================

required_columns = ["text", "sentiment"]

for column in required_columns:
    if column not in df.columns:
        raise Exception(
            f"Missing required column: {column}\n"
            f"Available columns: {list(df.columns)}"
        )

# Remove missing values
df = df.dropna(subset=["text", "sentiment"])

# Remove empty reviews
df = df[df["text"].str.strip() != ""]

# ==========================================================
# Features and Labels
# ==========================================================

X = df["text"]

y = df["sentiment"]

# ==========================================================
# Train/Test Split
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ==========================================================
# TF-IDF Vectorizer
# ==========================================================

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000
)

X_train_vectorized = vectorizer.fit_transform(X_train)

X_test_vectorized = vectorizer.transform(X_test)

# ==========================================================
# Train Model
# ==========================================================

print("=" * 60)
print("Training Sentiment Model...")
print("=" * 60)

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model.fit(
    X_train_vectorized,
    y_train
)

# ==========================================================
# Evaluation
# ==========================================================

prediction = model.predict(
    X_test_vectorized
)

accuracy = accuracy_score(
    y_test,
    prediction
)

print(f"Model Accuracy : {accuracy:.4f}")

# ==========================================================
# Save Model
# ==========================================================

joblib.dump(
    model,
    os.path.join(
        MODEL_DIR,
        "sentiment_model.pkl"
    )
)

joblib.dump(
    vectorizer,
    os.path.join(
        MODEL_DIR,
        "sentiment_vectorizer.pkl"
    )
)

# ==========================================================
# Finished
# ==========================================================

print("=" * 60)
print("Sentiment Model Saved Successfully!")
print("=" * 60)

print("Saved Files:")
print("✓ sentiment_model.pkl")
print("✓ sentiment_vectorizer.pkl")

print("=" * 60)
print("Training Completed Successfully")
print("=" * 60)