# train_model.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import joblib

# Load dataset
df = pd.read_csv("adult.csv")  # Make sure this file is in your directory

# Data preprocessing
df.dropna(inplace=True)

# Feature selection
categorical_cols = ["Gender", "Education Level", "Job Title"]
numerical_cols = ["Age", "Years of Experience"]
target_col = "Salary"

# Encode categorical features
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Prepare features and target
X = df[categorical_cols + numerical_cols]
y = df[target_col]

# Scale numerical features
scaler = StandardScaler()
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=4,
    random_state=42
)
model.fit(X_train, y_train)

# Save model artifacts
model_data = {
    "model": model,
    "label_encoders": label_encoders,
    "scaler": scaler,
    "categorical_cols": categorical_cols,
    "numerical_cols": numerical_cols,
    "target_col": target_col
}

joblib.dump(model_data, "salary_predictor.pkl")
print("Model saved successfully as salary_predictor.pkl")
