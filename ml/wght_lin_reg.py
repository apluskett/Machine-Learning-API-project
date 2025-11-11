"""
Weighted Linear Regression for F1 Predictions
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path

DATA_PATH = Path("data/f1_results.csv")
MODEL_PATH = Path("models/trained/linear_v0.1.joblib")

def load_data():
    print("📊 Loading data...")
    df = pd.read_csv(DATA_PATH)
    print(f"✅ Loaded {len(df):,} rows")
    return df

def create_features(df):
    print("🔧 Creating features...")
    
    # Convert driver to numeric
    if 'driver' in df.columns:
        driver_ids = df['driver'].astype('category').cat.codes
    else:
        driver_ids = range(len(df))
    
    X = pd.DataFrame({
        'grid': df['grid'],
        'driver_id': driver_ids,
        'points': df['points'] if 'points' in df.columns else 0
    })
    
    y = df['position']
    
    # Remove NaN
    mask = ~y.isna()
    X = X[mask]
    y = y[mask]
    
    print(f"✅ {len(X)} samples ready")
    return X, y

def create_weights(df):
    print("⚖️  Creating time weights...")
    years = df['year'].values
    max_year = years.max()
    weights = np.exp(-(max_year - years) * 0.1)
    print(f"✅ Weights: {weights.min():.3f} to {weights.max():.3f}")
    return weights

def train_model(X, y, weights):
    print("🎯 Training model...")
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = LinearRegression()
    model.fit(X_scaled, y, sample_weight=weights)
    
    score = model.score(X_scaled, y)
    print(f"✅ Trained! R² = {score:.4f}")
    
    return model, scaler

def save_model(model, scaler):
    print("💾 Saving model...")
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    package = {
        'model': model,
        'scaler': scaler,
        'version': 'v0.1'
    }
    
    joblib.dump(package, MODEL_PATH)
    print(f"✅ Saved to {MODEL_PATH}")

def main():
    print("=" * 60)
    print("F1 Weighted Linear Regression Training")
    print("=" * 60)
    print()
    
    df = load_data()
    X, y = create_features(df)
    weights = create_weights(df)
    model, scaler = train_model(X, y, weights)
    save_model(model, scaler)
    
    print("\n✅ Training complete!")
    print("Next: Update app/main.py and rebuild container")

if __name__ == "__main__":
    main()