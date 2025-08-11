# ✈️ Flight Fare Prediction Model

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.13-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/Scikit--learn-Model-orange?logo=scikit-learn" alt="Scikit-learn">
  <img src="https://img.shields.io/badge/Pandas-Data%20Analysis-yellow?logo=pandas" alt="Pandas">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="MIT License">
  <img src="https://img.shields.io/badge/Status-Completed-brightgreen" alt="Status">
</p>

---

## 🚀 Overview

The **Flight Fare Prediction Model** uses historical flight data to predict ticket prices based on airline, route, class, and travel details.  
Built with **Python** and **Scikit-learn**, it employs a **Random Forest Regressor** for high accuracy.

---

## ✨ Features

- 🛫 Predict airline ticket prices instantly
- 📊 Preprocessing with encoding, missing value handling, and scaling
- 🧠 Machine Learning with `RandomForestRegressor`
- 💾 Model & preprocessor saved using `joblib`
- 🔄 Ready-to-use prediction script

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/AnshGirap/flight-fare-prediction-model.git
cd flight-fare-prediction-model

# Install dependencies
pip install -r requirements.txt

# Train the model
python src/model.py

# Run prediction
python src/predict.py
📊 Example Prediction
python
Copy code
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
Metric	Score
MAE	₹1702.52
R² Score	0.979

🛠️ Tech Stack
Python

Pandas, NumPy

Scikit-learn

Joblib

Matplotlib / Seaborn

📜 License
This project is licensed under the MIT License.
```
