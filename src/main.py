from preprocessing import load_data, build_preprocessor
from model import train_model

# Step 1: Load and preprocess data
df = load_data()
preprocessor = build_preprocessor(df)

X = df.drop(columns=['price'])
y = df['price']

print("[INFO] Transforming data...")
X_processed = preprocessor.fit_transform(X)

# Step 2: Train model
print("[INFO] Starting training...")
model = train_model(X_processed, y)

# Step 3: Save artifacts
import joblib
joblib.dump(model, "../models/flight_fare_model.pkl")
joblib.dump(preprocessor, "../models/preprocessor.pkl")

print("[INFO] Training complete and files saved to ../models/")