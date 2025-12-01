from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import logging
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="F1 Prediction API",
    description="ML predictions for F1 2025 races using Weighted Linear Regression and XGBoost",
    version="1.0.0"
)

# Model paths
LINEAR_MODEL_PATH = Path("models/trained/linear_v0.1.joblib")
XGBOOST_MODEL_PATH = Path("models/trained/xgboost_v0.1.joblib")
TRAINING_DATA_PATH = Path("dataset/F1_training_data.csv")

# Global model storage
models = {
    "linear": None,
    "xgboost": None,
    "training_data": None
}

@app.on_event("startup")
async def startup():
    logger.info("🏎️  F1 API starting up...")
    
    # Load linear regression model
    try:
        if LINEAR_MODEL_PATH.exists():
            package = joblib.load(LINEAR_MODEL_PATH)
            models["linear"] = package
            logger.info(f"✅ Loaded Linear Regression model: {package['version']}")
        else:
            logger.warning(f"⚠️  Linear model not found at {LINEAR_MODEL_PATH}")
    except Exception as e:
        logger.error(f"❌ Failed to load Linear model: {e}")
    
    # Load XGBoost model (when ready)
    try:
        if XGBOOST_MODEL_PATH.exists():
            package = joblib.load(XGBOOST_MODEL_PATH)
            models["xgboost"] = package
            logger.info(f"✅ Loaded XGBoost model: {package['version']}")
        else:
            logger.info("ℹ️  XGBoost model not yet trained")
    except Exception as e:
        logger.error(f"❌ Failed to load XGBoost model: {e}")
    
    # Load training data for feature engineering
    try:
        if TRAINING_DATA_PATH.exists():
            models["training_data"] = pd.read_csv(TRAINING_DATA_PATH)
            logger.info(f"✅ Loaded training data: {len(models['training_data'])} rows")
        else:
            logger.warning(f"⚠️  Training data not found at {TRAINING_DATA_PATH}")
    except Exception as e:
        logger.error(f"❌ Failed to load training data: {e}")
    
    logger.info("✅ API startup complete")

# Request/Response Models
class PredictionRequest(BaseModel):
    driver: str
    grid: float
    track: str
    
class PredictionResponse(BaseModel):
    driver: str
    track: str
    grid: float
    predicted_position: float
    model: str
    
class RacePredictionRequest(BaseModel):
    track: str
    grid_positions: dict  # {"driver_name": grid_position}

@app.get("/")
def root():
    return {
        "message": "F1 Prediction API",
        "version": "1.0.0",
        "status": "ready",
        "models": {
            "linear": models["linear"] is not None,
            "xgboost": models["xgboost"] is not None
        },
        "endpoints": {
            "predict_single": "/predict/{model}",
            "predict_race": "/predict-race/{model}",
            "models_info": "/models",
            "health": "/health"
        }
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "models_loaded": {
            "linear": models["linear"] is not None,
            "xgboost": models["xgboost"] is not None
        }
    }

@app.get("/dependencies")
def check_dependencies():
    """Check that all ML dependencies are installed"""
    deps = {}
    
    try:
        import fastapi
        deps["fastapi"] = fastapi.__version__
    except:
        deps["fastapi"] = "NOT INSTALLED"
    
    try:
        import sklearn
        deps["scikit-learn"] = sklearn.__version__
    except:
        deps["scikit-learn"] = "NOT INSTALLED"
    
    try:
        import xgboost
        deps["xgboost"] = xgboost.__version__
    except:
        deps["xgboost"] = "NOT INSTALLED"
    
    try:
        import pandas
        deps["pandas"] = pandas.__version__
    except:
        deps["pandas"] = "NOT INSTALLED"
    
    try:
        import numpy
        deps["numpy"] = numpy.__version__
    except:
        deps["numpy"] = "NOT INSTALLED"
    
    try:
        import fastf1
        deps["fastf1"] = fastf1.__version__
    except:
        deps["fastf1"] = "NOT INSTALLED"
    
    return {
        "status": "all dependencies checked",
        "dependencies": deps
    }

@app.get("/models")
def get_models_info():
    """Get information about loaded models"""
    info = {}
    
    if models["linear"]:
        info["linear"] = {
            "loaded": True,
            "version": models["linear"].get("version", "unknown"),
            "type": "Weighted Linear Regression"
        }
    else:
        info["linear"] = {"loaded": False}
    
    if models["xgboost"]:
        info["xgboost"] = {
            "loaded": True,
            "version": models["xgboost"].get("version", "unknown"),
            "type": "Weighted XGBoost"
        }
    else:
        info["xgboost"] = {"loaded": False}
    
    return info

def engineer_features_for_api(driver: str, grid: float, track: str, training_data: pd.DataFrame):
    """Engineer features for a single prediction"""
    
    # Calculate driver statistics from training data
    driver_data = training_data[training_data['driver'] == driver]
    
    if len(driver_data) > 0:
        driver_avg_position = driver_data['position'].mean()
        driver_avg_points = driver_data['points'].mean()
        driver_recent_form = driver_data.tail(3)['position'].mean()
        
        # Driver-track specific
        driver_track_data = driver_data[driver_data['track'] == track]
        if len(driver_track_data) > 0:
            driver_track_avg = driver_track_data['position'].mean()
        else:
            driver_track_avg = driver_avg_position
    else:
        # New driver - use defaults
        driver_avg_position = 10.0
        driver_avg_points = 5.0
        driver_recent_form = 10.0
        driver_track_avg = 10.0
    
    # Track statistics
    track_data = training_data[training_data['track'] == track]
    if len(track_data) > 0:
        track_avg_pos_change = (track_data['grid'] - track_data['position']).mean()
    else:
        track_avg_pos_change = 0.0
    
    features = {
        'grid': grid,
        'driver_avg_position': driver_avg_position,
        'driver_avg_points': driver_avg_points,
        'driver_recent_form': driver_recent_form,
        'track_avg_pos_change': track_avg_pos_change,
        'driver_track_avg': driver_track_avg
    }
    
    return features

@app.post("/predict/{model_name}", response_model=PredictionResponse)
def predict_single(model_name: str, request: PredictionRequest):
    """
    Predict race finish position for a single driver
    
    - **model_name**: 'linear' or 'xgboost'
    - **driver**: Driver name (e.g., "Max Verstappen")
    - **grid**: Starting grid position (1-20)
    - **track**: Track name (e.g., "Monaco")
    """
    
    # Validate model
    if model_name not in ["linear", "xgboost"]:
        raise HTTPException(status_code=400, detail="Model must be 'linear' or 'xgboost'")
    
    if models[model_name] is None:
        raise HTTPException(status_code=503, detail=f"{model_name} model not loaded")
    
    if models["training_data"] is None:
        raise HTTPException(status_code=503, detail="Training data not loaded")
    
    try:
        # Engineer features
        features = engineer_features_for_api(
            request.driver, 
            request.grid, 
            request.track,
            models["training_data"]
        )
        
        # Create feature vector
        feature_cols = ['grid', 'driver_avg_position', 'driver_avg_points', 
                       'driver_recent_form', 'track_avg_pos_change', 'driver_track_avg']
        X = pd.DataFrame([features])[feature_cols]
        
        # Scale and predict
        model_package = models[model_name]
        scaler = model_package['scaler']
        model = model_package['model']
        
        X_scaled = scaler.transform(X)
        prediction = model.predict(X_scaled)[0]
        
        # Clip to valid range
        prediction = float(np.clip(prediction, 1, 20))
        
        return PredictionResponse(
            driver=request.driver,
            track=request.track,
            grid=request.grid,
            predicted_position=prediction,
            model=model_name
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict-race/{model_name}")
def predict_race(model_name: str, request: RacePredictionRequest):
    """
    Predict finish positions for all drivers in a race
    
    - **model_name**: 'linear' or 'xgboost'
    - **track**: Track name
    - **grid_positions**: Dictionary of driver names to grid positions
    
    Returns predictions sorted by predicted finish position
    """
    
    # Validate model
    if model_name not in ["linear", "xgboost"]:
        raise HTTPException(status_code=400, detail="Model must be 'linear' or 'xgboost'")
    
    if models[model_name] is None:
        raise HTTPException(status_code=503, detail=f"{model_name} model not loaded")
    
    try:
        predictions = []
        
        for driver, grid_pos in request.grid_positions.items():
            pred_request = PredictionRequest(
                driver=driver,
                grid=float(grid_pos),
                track=request.track
            )
            
            result = predict_single(model_name, pred_request)
            predictions.append(result.dict())
        
        # Sort by predicted position
        predictions.sort(key=lambda x: x['predicted_position'])
        
        return {
            "track": request.track,
            "model": model_name,
            "predictions": predictions
        }
        
    except Exception as e:
        logger.error(f"Race prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Race prediction failed: {str(e)}")