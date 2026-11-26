import pandas as pd
import pickle
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score

# Project folder location
BASE_DIR = Path(__file__).resolve().parent

# Load dataset
data = pd.read_csv(BASE_DIR / "car data.csv")

print("Dataset Loaded Successfully!")
print("Dataset Shape:", data.shape)

# Input features
X = data[
    [
        'Year',
        'Present_Price',
        'Kms_Driven',
        'Fuel_Type',
        'Seller_Type',
        'Transmission',
        'Owner'
    ]
]

# Target variable
y = data['Selling_Price']

# Categorical columns
categorical = [
    'Fuel_Type',
    'Seller_Type',
    'Transmission'
]

# Numerical columns
numerical = [
    'Year',
    'Present_Price',
    'Kms_Driven',
    'Owner'
]

# Data preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical),
        ('num', 'passthrough', numerical)
    ]
)

# Random Forest model
model = Pipeline(
    steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(
            n_estimators=100,
            random_state=42
        ))
    ]
)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
print("\nTraining Machine Learning Model...")

model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Model evaluation
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\n----- MODEL EVALUATION -----")
print("Mean Absolute Error:", round(mae, 3))
print("R2 Score:", round(r2, 3))

# Save trained model
model_path = BASE_DIR / "car_price_model.pkl"

with open(model_path, "wb") as file:
    pickle.dump(model, file)

print("\nModel trained successfully!")
print("Model saved as car_price_model.pkl")