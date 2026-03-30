# 🏎️ F1 Podium Predictor API

A machine learning microservice for predicting Formula 1 race finishing positions using historical data from 2018-2025. Built with Python, FastAPI, scikit-learn, and XGBoost. Consumed by the [F1 Podium Predictor Web Application](#).

---

## 📊 Overview

Predicts F1 race finishing positions based on:
- Starting grid position
- Driver historical performance
- Track characteristics
- Recent form and momentum

### Architecture
```
┌─────────────────────┐      ┌──────────────────────┐      ┌─────────────────────┐
│  React Frontend     │ ───> │  Rails API Backend   │ ───> │  ML Prediction API  │
│  (Port 3001)        │      │  (Port 3000)         │      │  (FastAPI:8000)     │
│  • Race prediction  │      │  • Prediction storage│      │  • Linear Regression│
│  • Results display  │      │  • API proxy         │      │  • XGBoost          │
│  • Model comparison │      │  • Database          │      │  • Feature engineer │
└─────────────────────┘      └──────────────────────┘      └─────────────────────┘
```

---

## 📈 Model Performance

**Training Data**: 2,905 races (2018-2025 seasons)

| Model | MAE | RMSE |
|-------|-----|------|
| **Linear Regression** | 2.49 | 3.28 |
| **XGBoost** | 2.37 | 3.01 |

**Qatar GP 2025 Validation**:
- ✅ Verstappen P1: Predicted 2.05 → Actual P1
- ✅ Piastri P2: Predicted 3.71 → Actual P2
- ✅ Russell P6: Predicted 6.92 → Actual P6
- ✅ Alonso P7: Predicted 7.61 → Actual P7

---

## 🔧 Feature Engineering

Instead of categorical encoding, 6 numerical features were engineered:

| Feature | Description |
|---------|-------------|
| `grid` | Starting grid position (1-20) |
| `driver_avg_position` | Driver's rolling 5-race average finish |
| `driver_avg_points` | Driver's rolling 5-race average points |
| `driver_recent_form` | Weighted last 3 races (3x, 2x, 1x) |
| `track_avg_pos_change` | Track's average positions gained/lost |
| `driver_track_avg` | Driver's historical performance at specific track |

### Time-Based Weighting
```python
weight = exp(-(max_year - race_year) * 0.22)
```
- 2018 races: weight ≈ 0.21
- 2025 races: weight = 1.00

---

## 🚀 Quick Start
```bash
# Build container
podman build -t f1-prediction-api .

# Run container
podman run -d --name f1-api -p 8000:8000 f1-prediction-api

# Verify
curl http://localhost:8000/health
```

---

## 📚 API Documentation

**Interactive Docs**: http://localhost:8000/docs

### Endpoints

#### Single Driver Prediction
```bash
POST /predict/{model}

{
  "driver": "Max Verstappen",
  "grid": 2,
  "track": "Qatar"
}
```

#### Full Race Prediction
```bash
POST /predict-race/{model}

{
  "track": "Abu Dhabi",
  "grid_positions": {
    "Max Verstappen": 1,
    "Lando Norris": 2,
    "Oscar Piastri": 3
  }
}
```

---

## 📁 Project Structure
```
f1-podium-predictor-api/
├── app/
│   └── main.py
├── ml/
│   ├── wght_lin_reg.py
│   ├── wght_xgboost.py
│   └── predict_remaining_races.py
├── scripts/
│   └── merge_script
├── dataset/
├── models/
│   └── trained/
├── Dockerfile
└── requirements.txt
```

---

## 📦 Dependencies
```
fastapi==0.104.1
uvicorn[standard]==0.24.0
scikit-learn==1.3.2
xgboost==2.0.3
pandas==2.1.3
numpy==1.26.2
joblib==1.3.2
pydantic==2.5.0
```

---

## 👤 Author

**Alex Pluskett**
[GitHub](https://github.com/apluskett)

---

## 📝 License

MIT License
