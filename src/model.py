import time
from sklearn.ensemble import RandomForestRegressor

def train_model(X, y):
    print("[INFO] Starting model training...")
    start_time = time.time()

    model = RandomForestRegressor(n_estimators=50, max_depth=15, random_state=42, n_jobs=-1)
    model.fit(X, y)

    print(f"[INFO] Training complete in {time.time() - start_time:.2f} seconds.")
    return model
