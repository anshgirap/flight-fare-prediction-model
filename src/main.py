import os
import pandas as pd
import joblib
from preprocessing import preprocess_data
from model import train_model

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "../data/airlines_flights_data.csv")
MODEL_DIR = os.path.join(BASE_DIR, "../models")
MODEL_PATH = os.path.join(MODEL_DIR, "flight_fare_model.pkl")
PREPROCESSOR_PATH = os.path.join(MODEL_DIR, "preprocessor.pkl")

# Ensure models directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# Load Dataset
print(f"[INFO] Loading dataset from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
print(f"[INFO] Dataset loaded. Shape: {df.shape}")

# Preprocess Data
print("[INFO] Preprocessing data...")
X, y, preprocessor = preprocess_data(df)
print(f"[INFO] Data preprocessing complete. Shape: {X.shape}")

# Train Model
print("[INFO] Training model...")
model = train_model(X, y, preprocessor)
print("[INFO] Model training complete.")

# Save Artifacts
joblib.dump(model, MODEL_PATH)
joblib.dump(preprocessor, PREPROCESSOR_PATH)
print(f"[INFO] Model saved to: {MODEL_PATH}")
print(f"[INFO] Preprocessor saved to: {PREPROCESSOR_PATH}")