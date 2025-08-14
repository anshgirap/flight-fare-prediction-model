import os
import pandas as pd
import joblib

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "../models")
MODEL_PATH = os.path.join(MODEL_DIR, "flight_fare_model.pkl")
PREPROCESSOR_PATH = os.path.join(MODEL_DIR, "preprocessor.pkl")

# Load Artifacts
model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)

# Input Data (example)
example_input = pd.DataFrame([{
    "index": 0,
    "airline": "Air_India",
    "flight": "AI-202",
    "source_city": "Delhi",
    "departure_time": "Morning",
    "stops": "one",
    "arrival_time": "Night",
    "destination_city": "Mumbai",
    "class": "Economy",
    "duration": 120,
    "days_left": 25
}])

# Fix missing columns if any
missing_cols = set(preprocessor.feature_names_in_) - set(example_input.columns)
for col in missing_cols:
    example_input[col] = 0  # default value for missing columns

# Reorder columns to match preprocessor
example_input = example_input[preprocessor.feature_names_in_]

# Transform & Predict
X_processed = preprocessor.transform(example_input)
predicted_fare = model.predict(X_processed)

print(f"Predicted Fare: ₹{predicted_fare[0]:.2f}")