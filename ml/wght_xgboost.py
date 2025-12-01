"""
Weighted XGBoost for F1 Predictions
Uses the same features and weighting as linear regression for fair comparison
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from pathlib import Path

DATA_PATH = Path("dataset/F1_training_data.csv")
MODEL_PATH = Path("models/trained/xgboost_v0.1.joblib")

def load_data():
    print("📊 Loading training data...")
    df = pd.read_csv(DATA_PATH)
    print(f"✅ Loaded {len(df):,} rows")
    print(f"   Years: {int(df['year'].min())} - {int(df['year'].max())}")
    print(f"   Columns: {list(df.columns)}")
    return df

def create_features(df):
    print("\n🔧 Creating feature matrix...")
    
    # Use SAME features as linear regression for fair comparison
    feature_cols = [
        'grid',                    # Starting position
        'driver_avg_position',     # Driver's rolling avg finish
        'driver_avg_points',       # Driver's rolling avg points
        'driver_recent_form',      # Driver's weighted recent form
        'track_avg_pos_change',    # Track overtaking metric
        'driver_track_avg'         # Driver's performance at this track
    ]
    
    print(f"   Using features: {feature_cols}")
    
    X = df[feature_cols].copy()
    y = df['position'].copy()
    
    # Remove any rows with NaN
    mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[mask]
    y = y[mask]
    
    print(f"✅ Feature matrix ready: {len(X)} samples x {len(feature_cols)} features")
    print(f"   Target range: {y.min():.0f} - {y.max():.0f}")
    
    return X, y, df[mask]

def create_weights(df):
    print("\n⚖️  Creating time-based weights (recent races matter more)...")
    years = df['year'].values
    max_year = years.max()
    
    # SAME decay rate as linear regression for fair comparison
    decay_rate = 0.22
    weights = np.exp(-(max_year - years) * decay_rate)
    
    print(f"   Weight range: {weights.min():.3f} to {weights.max():.3f}")
    print(f"   2018 weight: {weights[years == 2018].mean():.3f}")
    print(f"   2025 weight: {weights[years == 2025].mean():.3f}")
    
    return weights

def train_model(X, y, weights):
    print("\n🎯 Training weighted XGBoost...")
    
    # Split data for validation (same as linear regression)
    X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(
        X, y, weights, test_size=0.15, random_state=42
    )
    
    print(f"   Training samples: {len(X_train)}")
    print(f"   Validation samples: {len(X_val)}")
    
    # Scale features (keeps comparison fair with linear regression)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # XGBoost parameters optimized for regression
    params = {
        'objective': 'reg:squarederror',
        'max_depth': 6,              # Depth of trees
        'learning_rate': 0.1,        # Step size shrinkage
        'n_estimators': 200,         # Number of boosting rounds
        'min_child_weight': 1,       # Minimum sum of instance weight
        'subsample': 0.8,            # Subsample ratio of training instances
        'colsample_bytree': 0.8,     # Subsample ratio of columns
        'gamma': 0,                  # Minimum loss reduction
        'reg_alpha': 0.1,            # L1 regularization
        'reg_lambda': 1.0,           # L2 regularization
        'random_state': 42,
        'n_jobs': -1                 # Use all CPU cores
    }
    
    print(f"\n🌲 XGBoost Parameters:")
    for key, val in params.items():
        print(f"   {key:20s}: {val}")
    
    # Train XGBoost with sample weights
    model = xgb.XGBRegressor(**params)
    
    model.fit(
        X_train_scaled, 
        y_train, 
        sample_weight=w_train,
        eval_set=[(X_val_scaled, y_val)],
        sample_weight_eval_set=[w_val],
        verbose=False
    )
    
    # Evaluate
    train_pred = model.predict(X_train_scaled)
    val_pred = model.predict(X_val_scaled)
    
    train_r2 = r2_score(y_train, train_pred)
    val_r2 = r2_score(y_val, val_pred)
    
    mae = mean_absolute_error(y_val, val_pred)
    rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    
    print(f"\n📊 Model Performance:")
    print(f"   Training R²: {train_r2:.4f}")
    print(f"   Validation R²: {val_r2:.4f}")
    print(f"   Validation MAE: {mae:.2f} positions")
    print(f"   Validation RMSE: {rmse:.2f} positions")
    
    # Show feature importance
    print(f"\n📈 Feature Importance (gain):")
    importance = model.get_booster().get_score(importance_type='gain')
    
    # Map feature names (XGBoost uses f0, f1, etc.)
    feature_names = X.columns.tolist()
    for i, feat in enumerate(feature_names):
        feat_key = f'f{i}'
        if feat_key in importance:
            print(f"   {feat:25s}: {importance[feat_key]:.2f}")
    
    return model, scaler

def save_model(model, scaler):
    print("\n💾 Saving model...")
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    package = {
        'model': model,
        'scaler': scaler,
        'version': 'v0.1',
        'model_type': 'xgboost'
    }
    
    joblib.dump(package, MODEL_PATH)
    print(f"✅ Saved to {MODEL_PATH}")

def main():
    print("=" * 70)
    print("F1 Weighted XGBoost Training")
    print("=" * 70)
    
    df = load_data()
    X, y, df_clean = create_features(df)
    weights = create_weights(df_clean)
    model, scaler = train_model(X, y, weights)
    save_model(model, scaler)
    
    print("\n" + "=" * 70)
    print("✅ Training complete!")
    print("=" * 70)
    print("\n📝 Next steps:")
    print("   1. Model saved to: models/trained/xgboost_v0.1.joblib")
    print("   2. Compare with linear regression performance")
    print("   3. Test predictions with scripts/predict_remaining_races.py")
    print("   4. XGBoost is now available in the API at /predict/xgboost")

if __name__ == "__main__":
    main()
