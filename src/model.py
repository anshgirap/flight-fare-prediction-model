import time
import os
import joblib
from sklearn.ensemble import RandomForestRegressor

# Adjust paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "flight_fare_pipeline.pkl")

def train_model(X, y, preprocessor):
    print("[INFO] Starting model training...")
    start_time = time.time()

    model = RandomForestRegressor(
        n_estimators=50,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, y)

    print(f"[INFO] Training complete in {time.time() - start_time:.2f} seconds.")

    # Save both model and preprocessor together
    pipeline_data = {
        "model": model,
        "preprocessor": preprocessor
    }
    joblib.dump(pipeline_data, MODEL_PATH)
    print(f"[INFO] Saved pipeline to {MODEL_PATH}")

    return model