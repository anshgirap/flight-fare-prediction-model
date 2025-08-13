import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from pathlib import Path

def load_data():
    csv_path = Path(__file__).resolve().parent.parent / "data" / "airlines_flights_data.csv"
    print(f"[INFO] Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"[INFO] Dataset loaded. Shape: {df.shape}")
    return df

def build_preprocessor(df):
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = df.select_dtypes(exclude=["object"]).drop(columns=["fare"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
        ]
    )
    return preprocessor