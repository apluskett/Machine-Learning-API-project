from fastapi import FastAPI
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="F1 Prediction API",
    description="ML predictions for F1 2025 races",
    version="1.0.0"
)

@app.on_event("startup")
async def startup():
    logger.info("🏎️  F1 API starting up...")
    logger.info("✅ All dependencies loaded")

@app.get("/")
def root():
    return {
        "message": "F1 Prediction API",
        "version": "1.0.0",
        "status": "ready"
    }

@app.get("/health")
def health():
    return {"status": "healthy"}

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