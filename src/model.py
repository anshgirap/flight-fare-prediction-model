from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import time

def train_model(X, y):
    model = RandomForestRegressor(
        n_estimators=50,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )

    print("[INFO] Starting model training...")
    start_time = time.time()

    model.fit(X, y)

    print(f"[INFO] Training complete in {time.time() - start_time:.2f} seconds.")
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"R² Score: {r2:.2f}")