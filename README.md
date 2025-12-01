# 🏎️ F1 Race Prediction API & Web Application

A full-stack machine learning system for predicting Formula 1 race finishing positions using historical data from 2018-2025. Built with Python (FastAPI + scikit-learn + XGBoost), Ruby on Rails, and React.

**MSU Denver - Machine Learning & Web Development Final Project**

---

## 📊 Project Overview

This project predicts F1 race finishing positions based on:
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

**🔗 Related Repository**: [Web Application Frontend/Backend](https://github.com/apluskett/final-web-app-assignments)  
This microservice API is consumed by the full-stack Rails + React application.

---

## 🎯 Features

### Machine Learning
- **Two Models**: Weighted Linear Regression & XGBoost
- **Time-weighted Training**: Recent seasons weighted more heavily (exponential decay)
- **Feature Engineering**: 6 engineered features replace categorical variables
- **Real-time Predictions**: Full race predictions via REST API

### Web Application
- **Interactive UI**: Select track, configure grid positions, choose model
- **Live Results**: Compare predictions vs actual race results
- **Model Comparison**: Side-by-side Linear vs XGBoost accuracy
- **Historical Data**: View past predictions and performance stats

### Containerized Deployment
- **Dockerized ML API**: Podman/Docker container with models pre-loaded
- **Production Ready**: Health checks, proper error handling, CORS configured

---

## 📈 Model Performance

**Training Data**: 2,905 races (2018-2025 seasons, races 1-18)  
**Validation Set**: Races 19-24 (US, Mexico, Brazil, Las Vegas, Qatar, Abu Dhabi)

### Results on Real 2025 Races (5 races evaluated):

| Model | MAE | RMSE | Best For |
|-------|-----|------|----------|
| **Linear Regression** | 2.49 | 3.28 | Consistent predictions, top 10 |
| **XGBoost** | 2.37 | 3.01 | Overall accuracy, complex patterns |

**Qatar GP 2025 Example**:
- ✅ Verstappen P1: Predicted 2.05 → Actual P1 (Linear)
- ✅ Piastri P2: Predicted 3.71 → Actual P2
- ✅ Russell P6: Predicted 6.92 → Actual P6
- ✅ Alonso P7: Predicted 7.61 → Actual P7

---

## 🚀 Quick Start

### Prerequisites
```bash
# Python 3.10+
python --version

# Podman or Docker
podman --version

# Ruby 3.x & Rails (for web app)
ruby --version
rails --version

# Node.js 18+ (for React frontend)
node --version
```

### 1. Start ML Prediction API

```bash
# Build container
podman build -t f1-prediction-api .

# Run container
podman run -d --name f1-api -p 8000:8000 f1-prediction-api

# Verify it's running
curl http://localhost:8000/health
```

### 2. Train/Update Models (Optional)

```bash
# Regenerate training data from CSVs
python3 scripts/merge_script

# Train linear regression model
python3 ml/wght_lin_reg.py

# Train XGBoost model
python3 ml/wght_xgboost.py

# Generate predictions for races 19-24
python3 ml/predict_remaining_races.py linear
python3 ml/predict_remaining_races.py xgboost
```

### 3. Start Web Application

The web application (Rails + React) is in a separate repository:

**🔗 [F1 Prediction Web App](https://github.com/apluskett/final-web-app-assignments)**

```bash
# Clone the web app repository
git clone https://github.com/apluskett/final-web-app-assignments.git
cd final-web-app-assignments

# Follow setup instructions in that repository
# The web app will connect to this ML API at http://localhost:8000
```

---

## 📚 API Documentation

### Base URL
```
http://localhost:8000
```

### Interactive Docs
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Endpoints

#### Health Check
```bash
GET /health
```
Response:
```json
{
  "status": "healthy",
  "models_loaded": {
    "linear": true,
    "xgboost": true
  }
}
```

#### Single Driver Prediction
```bash
POST /predict/{model}
Content-Type: application/json

{
  "driver": "Max Verstappen",
  "grid": 2,
  "track": "Qatar"
}
```
Response:
```json
{
  "driver": "Max Verstappen",
  "track": "Qatar",
  "grid": 2.0,
  "predicted_position": 1.81,
  "model": "xgboost"
}
```

#### Full Race Prediction
```bash
POST /predict-race/{model}
Content-Type: application/json

{
  "track": "Abu Dhabi",
  "grid_positions": {
    "Max Verstappen": 1,
    "Lando Norris": 2,
    "Oscar Piastri": 3,
    "Charles Leclerc": 4,
    "Carlos Sainz": 5
  }
}
```

Response:
```json
{
  "track": "Abu Dhabi",
  "model": "linear",
  "predictions": [
    {
      "driver": "Max Verstappen",
      "grid": 1,
      "predicted_position": 2.54
    },
    ...
  ]
}
```

---

## 🔧 Technical Details

### Feature Engineering

Instead of categorical encoding for drivers/tracks, we engineered 6 numerical features:

| Feature | Description |
|---------|-------------|
| `grid` | Starting grid position (1-20) |
| `driver_avg_position` | Driver's rolling 5-race average finish |
| `driver_avg_points` | Driver's rolling 5-race average points |
| `driver_recent_form` | Weighted last 3 races (3x, 2x, 1x) |
| `track_avg_pos_change` | Track's average positions gained/lost |
| `driver_track_avg` | Driver's historical performance at specific track |

### Time-Based Weighting

Training samples weighted by exponential decay:
```python
weight = exp(-(max_year - race_year) * 0.22)
```
- 2018 races: weight ≈ 0.21
- 2025 races: weight = 1.00

### Models

**Linear Regression**:
- Weighted least squares
- StandardScaler normalization
- Training R²: 0.6259, Validation R²: 0.6382

**XGBoost**:
- Weighted regression trees
- max_depth=6, n_estimators=200
- L1/L2 regularization (alpha=0.1, lambda=1.0)
- Training R²: 0.8869, Validation R²: 0.5890

---

## 📁 Project Structure

```
Machine-Learning-API-project/
├── app/                          # FastAPI application
│   ├── main.py                   # API routes & endpoints
│   └── __init__.py
├── ml/                           # Machine learning scripts
│   ├── wght_lin_reg.py          # Linear regression training
│   ├── wght_xgboost.py          # XGBoost training
│   ├── predict_remaining_races.py # Prediction pipeline
│   └── compare_models.py        # Model comparison
├── scripts/
│   └── merge_script             # Data preprocessing & feature engineering
├── dataset/                      # CSV data files
│   ├── F1_training_data.csv     # Processed training data (2,905 rows)
│   ├── Formula1_2018Season_RaceResults.csv
│   ├── ...                      # 2019-2024 seasons
│   └── Formula1_2025Season_RaceResults.csv
├── models/
│   └── trained/                 # Saved model files
│       ├── linear_v0.1.joblib
│       └── xgboost_v0.1.joblib
├── predictions/                  # Prediction outputs
│   ├── 2025_races_19-24_predictions_linear.csv
│   └── 2025_races_19-24_predictions_xgboost.csv
├── Dockerfile                    # Container configuration
├── requirements.txt              # Python dependencies
├── DEPLOYMENT.md                 # Deployment guide
└── README.md                     # This file
```

---

## 🧪 Testing & Validation

### Test the API
```bash
# Health check
curl http://localhost:8000/health

# Single prediction
curl -X POST "http://localhost:8000/predict/linear" \
  -H "Content-Type: application/json" \
  -d '{"driver": "Max Verstappen", "grid": 1, "track": "Qatar"}'

# Race prediction
curl -X POST "http://localhost:8000/predict-race/xgboost" \
  -H "Content-Type: application/json" \
  -d '{
    "track": "Abu Dhabi",
    "grid_positions": {
      "Max Verstappen": 1,
      "Lando Norris": 2,
      "Oscar Piastri": 3
    }
  }'
```

### Compare Models
```bash
python3 ml/compare_models.py
```

---

## 📦 Dependencies

### Python (ML API)
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

### System Requirements
- **CPU**: 2+ cores recommended
- **RAM**: 2GB minimum (models + data in memory)
- **Storage**: ~500MB for container + data

---

## 🚢 Deployment

### Update Models After New Races

```bash
# 1. Add new race data to CSV
# Edit dataset/Formula1_2025Season_RaceResults.csv

# 2. Regenerate training data
python3 scripts/merge_script

# 3. Retrain models
python3 ml/wght_lin_reg.py
python3 ml/wght_xgboost.py

# 4. Rebuild container
podman stop f1-api && podman rm f1-api
podman build -t f1-prediction-api .
podman run -d --name f1-api -p 8000:8000 f1-prediction-api

# 5. Generate new predictions
python3 ml/predict_remaining_races.py linear
python3 ml/predict_remaining_races.py xgboost
```

### Production Deployment
See `DEPLOYMENT.md` for full production deployment guide including:
- Port configuration
- Auto-restart setup
- Environment variables
- Troubleshooting

---

## 📊 Data Sources

- **F1 Historical Data**: 2018-2025 race results
- **Features**: Grid positions, race results, points, fastest laps
- **Drivers**: 2025 season lineup (20 drivers)
- **Tracks**: 24 circuits (2025 calendar)

---

## 🎓 Academic Context

**Course**: Machine Learning & Web Development  
**Institution**: Metropolitan State University of Denver  
**Semester**: Fall 2025

### Project Structure
This project consists of two repositories:
1. **[Machine-Learning-API-project](https://github.com/apluskett/Machine-Learning-API-project)** (this repo) - ML microservice
2. **[final-web-app-assignments](https://github.com/apluskett/final-web-app-assignments)** - Web application

### Learning Objectives Demonstrated
1. ✅ Machine learning model development (regression)
2. ✅ Feature engineering for categorical data
3. ✅ Time-series weighting strategies
4. ✅ Model comparison and validation
5. ✅ REST API development (FastAPI)
6. ✅ Containerization (Docker/Podman)
7. ✅ Microservice architecture
8. ✅ Full-stack integration (Rails + React consuming ML API)
9. ✅ Real-world data processing
10. ✅ API design and documentation

---

## 🤝 Contributing

This is an academic project, but suggestions welcome!

1. Fork the repository
2. Create feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open Pull Request

---

## 📝 License

MIT License - see LICENSE file for details

---

## 👤 Author

**Andrew Pluskett**  
MSU Denver - Computer Science  
[GitHub](https://github.com/apluskett)

---

## 🙏 Acknowledgments

- MSU Denver Machine Learning Course
- F1 historical data sources
- FastAPI, scikit-learn, XGBoost communities
- Ruby on Rails & React ecosystems

---

## 📞 Support

- **Issues**: Open a GitHub issue
- **API Docs**: http://localhost:8000/docs
- **Deployment Guide**: See `DEPLOYMENT.md`

---

**Built with ❤️ for racing and machine learning**
