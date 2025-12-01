# F1 Prediction API - Deployment Guide

## Quick Start

### Building the Container
```bash
podman build -t f1-prediction-api .
```

### Running the Container
```bash
# Run in detached mode
podman run -d --name f1-api -p 8000:8000 f1-prediction-api

# Run with logs visible
podman run --name f1-api -p 8000:8000 f1-prediction-api
```

### Managing the Container
```bash
# Check status
podman ps

# View logs
podman logs f1-api

# Stop container
podman stop f1-api

# Remove container
podman rm f1-api

# Restart container
podman restart f1-api
```

## API Endpoints

### Health Check
```bash
curl http://localhost:8000/health
```

### Get Models Info
```bash
curl http://localhost:8000/models
```

### Single Driver Prediction
```bash
curl -X POST "http://localhost:8000/predict/linear" \
  -H "Content-Type: application/json" \
  -d '{
    "driver": "Max Verstappen",
    "grid": 2,
    "track": "Qatar"
  }'
```

### Full Race Prediction
```bash
curl -X POST "http://localhost:8000/predict-race/xgboost" \
  -H "Content-Type: application/json" \
  -d '{
    "track": "Abu Dhabi",
    "grid_positions": {
      "Max Verstappen": 1,
      "Lando Norris": 2,
      "Oscar Piastri": 3,
      "Charles Leclerc": 4,
      "Carlos Sainz": 5
    }
  }'
```

## Interactive API Documentation

Once running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Model Selection

The API supports two models:
- `linear` - Weighted Linear Regression (R² 0.6382, MAE 2.26)
- `xgboost` - Weighted XGBoost (R² 0.5890, MAE 2.45)

Replace `{model}` in endpoints with either `linear` or `xgboost`.

## Production Deployment

### Port Configuration
Change the port mapping:
```bash
podman run -d --name f1-api -p 80:8000 f1-prediction-api
```

### Auto-restart on System Boot
```bash
podman run -d --name f1-api --restart=always -p 8000:8000 f1-prediction-api
```

### Environment Variables (if needed)
```bash
podman run -d --name f1-api \
  -p 8000:8000 \
  -e LOG_LEVEL=info \
  f1-prediction-api
```

## Updating the Models

When you retrain models or update data:

1. Stop and remove the old container:
```bash
podman stop f1-api
podman rm f1-api
```

2. Rebuild the image:
```bash
podman build -t f1-prediction-api .
```

3. Run the new container:
```bash
podman run -d --name f1-api -p 8000:8000 f1-prediction-api
```

## Troubleshooting

### Check if container is running
```bash
podman ps -a
```

### View real-time logs
```bash
podman logs -f f1-api
```

### Enter container for debugging
```bash
podman exec -it f1-api /bin/bash
```

### Test from inside container
```bash
podman exec f1-api curl http://localhost:8000/health
```

## Performance Notes

- Models are loaded on startup (takes ~2-3 seconds)
- Both linear and xgboost models are kept in memory
- Training data (2,905 samples) is loaded for feature engineering
- Average response time: ~50-100ms per prediction
