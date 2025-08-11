import pandas as pd
import joblib

# Load artifacts
model = joblib.load("../models/flight_fare_model.pkl")
preprocessor = joblib.load("../models/preprocessor.pkl")

# Input
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

# Transform input
X_processed = preprocessor.transform(example_input)

# Predict
predicted_fare = model.predict(X_processed)
print(f"Predicted Fare: ₹{predicted_fare[0]:.2f}")