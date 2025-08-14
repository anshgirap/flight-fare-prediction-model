import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def load_data():
    csv_path = "../data/airlines_flights_data.csv"
    df = pd.read_csv(csv_path)
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

def preprocess_data(df):
    preprocessor = build_preprocessor(df)
    X = df.drop(columns=["fare"])
    y = df["fare"]
    X_processed = preprocessor.fit_transform(X)
    return X_processed, y, preprocessor