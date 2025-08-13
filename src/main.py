from preprocessing import load_data, build_preprocessor
from model import train_model
from pathlib import Path
import joblib

# Ensure models folder exists
models_dir = Path(__file__).resolve().parent.parent / "models"
models_dir.mkdir(parents=True, exist_ok=True)

# Load and preprocess
df = load_data()
preprocessor = build_preprocessor(df)

X = df.drop(columns=["fare"])
y = df["fare"]

print("[INFO] Transforming data...")
X_processed = preprocessor.fit_transform(X)

# Train
model = train_model(X_processed, y)

# Save model
model_path = models_dir / "flight_fare_model.pkl"
joblib.dump(model, model_path)
print(f"[INFO] Model saved to: {model_path}")