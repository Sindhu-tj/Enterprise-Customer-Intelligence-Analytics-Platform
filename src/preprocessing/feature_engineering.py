import pandas as pd

# Load dataset
df = pd.read_csv("data/raw/customer_churn.csv")

# Remove extra spaces from column names
df.columns = df.columns.str.strip()

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

# ===============================
# Feature Engineering
# ===============================

# Average Monthly Spend
df["AverageMonthlySpend"] = df["TotalCharges"] / (df["tenure"] + 1)

# Long-Term Customer
df["LongTermCustomer"] = (df["tenure"] >= 24).astype(int)

# High Monthly Charges
df["HighMonthlyCharges"] = (df["MonthlyCharges"] > df["MonthlyCharges"].median()).astype(int)

print("Feature Engineering Completed Successfully!")
print(df.head())

# Save engineered dataset
df.to_csv("data/processed/customer_churn_engineered.csv", index=False)

print("Engineered dataset saved successfully!")