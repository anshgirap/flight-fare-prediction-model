# ✈️ Flight Fare Prediction Model

<p align="center"> <img src="https://img.shields.io/badge/Python-3.13-blue?logo=python" alt="Python"> <img src="https://img.shields.io/badge/Scikit--learn-Model-orange?logo=scikit-learn" alt="Scikit-learn"> <img src="https://img.shields.io/badge/Pandas-Data%20Analysis-yellow?logo=pandas" alt="Pandas"> <img src="https://img.shields.io/badge/License-MIT-green" alt="MIT License"> <img src="https://img.shields.io/badge/Status-Completed-brightgreen" alt="Status"> </p>
Predict airline ticket prices with Machine Learning using a Random Forest Regressor trained on historical flight data.

---

## 🚀 Features

✅ Data Preprocessing – Encoding, missing value handling, and scaling

✅ Machine Learning Model – RandomForestRegressor for high accuracy

✅ Artifacts Saved – Model & preprocessor with joblib

✅ Instant Predictions – Pass flight details, get fare instantly

---

## ⚙️ Installation & Setup

### Clone repository

git clone https://github.com/anshgirap/flight-fare-prediction-model.git
cd flight-fare-prediction-model

### Install dependencies

pip install -r requirements.txt

### Train model

python src/model.py

### Run prediction

python src/predict.py

---

## 📊 Example Prediction

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

Predicted Fare: ₹6859.18
📈 Model Performance
Metric Score
MAE ₹1702.52
R² 0.979

## 🛠️ Tech Stack

Python, Pandas, NumPy

Scikit-learn, Joblib

## 📜 License

This project is licensed under the MIT License.
