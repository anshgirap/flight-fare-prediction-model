# ✈️ Flight Fare Prediction Model

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.13-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/Scikit--learn-Model-orange?logo=scikit-learn" alt="Scikit-learn">
  <img src="https://img.shields.io/badge/Pandas-Data%20Analysis-yellow?logo=pandas" alt="Pandas">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="MIT License">
  <img src="https://img.shields.io/badge/Status-Completed-brightgreen" alt="Status">
</p>

Predict airline ticket prices with **Machine Learning** using a **Random Forest Regressor** trained on historical flight data.

---

## 🌟 Features

- **Data Preprocessing:** Encoding categorical features, handling missing values, scaling numeric data
- **Machine Learning Model:** RandomForestRegressor for **high accuracy predictions**
- **Saved Artifacts:** Model & preprocessor stored via `joblib`
- **Instant Predictions:** Input flight details, get predicted fare instantly

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository

```bash
git clone https://github.com/anshgirap/flight-fare-prediction-model.git
cd flight-fare-prediction-model
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run main

```bash
python src/main.py
```

### 4️⃣ Run predictions

```bash
python src/predict.py
```

---

## 📊 Example Prediction

```python
import pandas as pd
from src.predict import predict_fare

example_input = pd.DataFrame([{
    'airline': 'Indigo',
    'flight': '6E-333',
    'source_city': 'Delhi',
    'departure_time': 'Morning',
    'stops': 'non-stop',
    'arrival_time': 'Afternoon',
    'destination_city': 'Mumbai',
    'class': 'Economy',
    'duration': 2.5,
    'days_left': 15
}])

fare = predict_fare(example_input)
print(f"Predicted Fare: ₹{fare:.2f}")
```

**Predicted Fare:** ₹6859.18

---

## 📈 Model Performance

| Metric | Score    |
| ------ | -------- |
| MAE    | ₹1702.52 |
| R²     | 0.979    |

---

## 🛠️ Tech Stack

- **Languages & Libraries:** Python, Pandas, NumPy
- **Machine Learning:** Scikit-learn, Joblib

---

## 📜 License

This project is licensed under the **MIT License**.
