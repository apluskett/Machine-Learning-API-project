"""
Predict the last 6 races of F1 2025 season (races 19-24)
- For completed races: compare predictions vs actual
- For future races: generate predictions based on historical data
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# Paths
TRAINING_DATA = Path("dataset/F1_training_data.csv")
FULL_2025_DATA = Path("dataset/Formula1_2025Season_RaceResults.csv")
LINEAR_MODEL_PATH = Path("models/trained/linear_v0.1.joblib")
XGBOOST_MODEL_PATH = Path("models/trained/xgboost_v0.1.joblib")
OUTPUT_DIR = Path("predictions")

def load_model(model_type="linear"):
    """Load the trained model"""
    print(f"📦 Loading {model_type} model...")
    
    if model_type == "xgboost":
        model_path = XGBOOST_MODEL_PATH
    else:
        model_path = LINEAR_MODEL_PATH
    
    if not model_path.exists():
        print(f"⚠️  {model_type} model not found at {model_path}")
        return None, None
    
    package = joblib.load(model_path)
    model = package['model']
    scaler = package['scaler']
    print(f"✅ Model loaded: {package['version']}")
    return model, scaler, model_type

def load_training_data():
    """Load training data to get driver/track statistics"""
    print("\n📊 Loading training data for feature engineering...")
    df = pd.read_csv(TRAINING_DATA)
    print(f"✅ Loaded {len(df)} training samples")
    return df

def load_2025_races():
    """Load full 2025 season data"""
    print("\n📊 Loading 2025 season data...")
    df = pd.read_csv(FULL_2025_DATA)
    
    # Standardize column names
    df = df.rename(columns={
        'Starting Grid': 'grid',
        'Position': 'position',
        'Points': 'points',
        'Driver': 'driver',
        'Track': 'track'
    })
    
    df['year'] = 2025
    
    print(f"✅ Loaded {len(df)} rows from 2025 season")
    
    return df

def get_last_6_races(df_2025):
    """Get the last 6 races (19-24) from 2025 data"""
    print("\n🏁 Identifying last 6 races...")
    
    unique_tracks = df_2025['track'].unique()
    print(f"   Total races in 2025: {len(unique_tracks)}")
    print(f"   All races: {list(unique_tracks)}")
    
    if len(unique_tracks) < 19:
        print(f"   ⚠️  Only {len(unique_tracks)} races available, need at least 19")
        last_6_tracks = unique_tracks[-6:] if len(unique_tracks) >= 6 else unique_tracks
    else:
        # Get exactly races 19-24 (0-indexed: positions 18-23)
        last_6_tracks = unique_tracks[18:24]
    
    print(f"\n   Last 6 races (19-24): {list(last_6_tracks)}")
    
    df_last_6 = df_2025[df_2025['track'].isin(last_6_tracks)].copy()
    
    # Convert position to numeric (handles 'NC', 'DNF', etc.)
    df_last_6['position'] = pd.to_numeric(df_last_6['position'], errors='coerce')
    df_last_6['grid'] = pd.to_numeric(df_last_6['grid'], errors='coerce')
    df_last_6['points'] = pd.to_numeric(df_last_6['points'], errors='coerce')
    
    # Identify which races have actual results vs need predictions
    completed_races = []
    future_races = []
    
    for track in last_6_tracks:
        track_data = df_last_6[df_last_6['track'] == track]
        if track_data['position'].notna().any():
            completed_races.append(track)
        else:
            future_races.append(track)
    
    print(f"\n   ✅ Completed races ({len(completed_races)}): {completed_races}")
    print(f"   🔮 Future races ({len(future_races)}): {future_races}")
    
    return df_last_6, last_6_tracks, completed_races, future_races

def fill_missing_races_with_historical(df_last_6, last_6_tracks, df_training):
    """
    For races not yet in 2025 data, create entries using:
    - Historical grid positions from prior seasons at same track
    - Current 2025 driver lineup
    """
    print("\n🔄 Filling missing races with historical grid data...")
    
    # Get current 2025 drivers
    drivers_2025 = df_training[df_training['year'] == 2025]['driver'].unique()
    
    if len(drivers_2025) == 0:
        # Fallback to most recent drivers
        drivers_2025 = df_training.groupby('driver').tail(1)['driver'].unique()[:20]
    
    print(f"   Current driver lineup: {len(drivers_2025)} drivers")
    
    # Expected last 6 tracks for 2025 season (races 19-24)
    # Based on actual 2025 F1 calendar
    expected_tracks = ['United States', 'Mexico', 'Brazil', 'Las Vegas', 'Qatar', 'Abu Dhabi']
    
    missing_tracks = []
    filled_data = []
    
    for track in expected_tracks:
        if track not in last_6_tracks:
            missing_tracks.append(track)
            
            # Get historical data for this track
            track_history = df_training[df_training['track'] == track].copy()
            
            if len(track_history) > 0:
                print(f"   📍 {track}: Using historical grid data")
                
                # Get most recent year's grid positions for this track
                recent_year = track_history['year'].max()
                recent_race = track_history[track_history['year'] == recent_year]
                
                # Create mapping of grid positions from recent race
                grid_positions = recent_race.set_index('driver')['grid'].to_dict()
                
                # Create entries for current 2025 drivers
                for driver in drivers_2025:
                    # Use historical grid if driver raced there, else estimate
                    if driver in grid_positions:
                        grid_pos = grid_positions[driver]
                    else:
                        # New driver - estimate based on their average position
                        driver_avg = df_training[df_training['driver'] == driver]['grid'].mean()
                        if pd.notna(driver_avg):
                            grid_pos = driver_avg
                        else:
                            grid_pos = 10.0  # Default middle of pack
                    
                    filled_data.append({
                        'year': 2025,
                        'track': track,
                        'driver': driver,
                        'grid': grid_pos,
                        'position': np.nan,  # Future race - no result yet
                        'points': np.nan
                    })
            else:
                print(f"   ⚠️  {track}: No historical data available, skipping")
    
    if filled_data:
        df_filled = pd.DataFrame(filled_data)
        print(f"   ✅ Created {len(df_filled)} entries for {len(missing_tracks)} missing races")
        
        # Combine with existing last 6 races
        df_combined = pd.concat([df_last_6, df_filled], ignore_index=True)
        
        # Update tracks list
        all_last_6_tracks = list(last_6_tracks) + missing_tracks
        
        return df_combined, all_last_6_tracks, missing_tracks
    else:
        print("   ℹ️  No missing races to fill")
        return df_last_6, list(last_6_tracks), []

def engineer_features_for_prediction(df_predict, df_training):
    """
    Engineer features for prediction races using training data statistics
    """
    print("\n🔧 Engineering features for prediction...")
    
    df_predict = df_predict.copy()
    
    # Calculate statistics from training data (up to race 18)
    driver_stats = df_training.groupby('driver').agg({
        'position': 'mean',
        'points': 'mean'
    }).rename(columns={
        'position': 'driver_avg_position',
        'points': 'driver_avg_points'
    })
    
    # Track statistics from training data
    track_stats = df_training.groupby('track').apply(
        lambda x: (x['grid'] - x['position']).mean()
    ).to_dict()
    
    # Driver-track combinations
    driver_track_stats = df_training.groupby(['driver', 'track'])['position'].mean().to_dict()
    
    # Merge driver stats
    df_predict = df_predict.merge(driver_stats, on='driver', how='left')
    
    # Add track avg position change
    df_predict['track_avg_pos_change'] = df_predict['track'].map(track_stats)
    
    # Add driver-track average
    df_predict['driver_track_avg'] = df_predict.apply(
        lambda row: driver_track_stats.get((row['driver'], row['track']), row['driver_avg_position']),
        axis=1
    )
    
    # For recent form, use last 3 races from training data per driver
    driver_recent = df_training.sort_values(['year', 'track']).groupby('driver').tail(3).groupby('driver')['position'].mean()
    df_predict['driver_recent_form'] = df_predict['driver'].map(driver_recent)
    
    # Fill missing values with driver averages
    df_predict['driver_recent_form'] = df_predict['driver_recent_form'].fillna(df_predict['driver_avg_position'])
    df_predict['driver_track_avg'] = df_predict['driver_track_avg'].fillna(df_predict['driver_avg_position'])
    
    # Fill any remaining NaN with medians
    for col in ['driver_avg_position', 'driver_avg_points', 'driver_recent_form', 
                'track_avg_pos_change', 'driver_track_avg']:
        if col in df_predict.columns:
            df_predict[col] = df_predict[col].fillna(df_predict[col].median())
    
    print(f"✅ Features engineered for {len(df_predict)} entries")
    
    return df_predict

def make_predictions(df, model, scaler):
    """Make predictions using the trained model"""
    print("\n🔮 Making predictions...")
    
    feature_cols = [
        'grid',
        'driver_avg_position',
        'driver_avg_points',
        'driver_recent_form',
        'track_avg_pos_change',
        'driver_track_avg'
    ]
    
    X = df[feature_cols].copy()
    X_scaled = scaler.transform(X)
    
    predictions = model.predict(X_scaled)
    
    # Clip predictions to valid range [1, 20]
    predictions = np.clip(predictions, 1, 20)
    
    df['predicted_position'] = predictions
    
    print(f"✅ Predictions complete")
    
    return df

def evaluate_predictions(df, completed_races):
    """Evaluate predictions for completed races"""
    if not completed_races:
        print("\n⚠️  No completed races to evaluate")
        return None
    
    print("\n📊 Evaluating predictions on completed races...")
    
    df_completed = df[df['track'].isin(completed_races) & df['position'].notna()].copy()
    
    if len(df_completed) == 0:
        print("   No data with actual results")
        return None
    
    mae = np.mean(np.abs(df_completed['predicted_position'] - df_completed['position']))
    rmse = np.sqrt(np.mean((df_completed['predicted_position'] - df_completed['position'])**2))
    
    print(f"\n   MAE: {mae:.2f} positions")
    print(f"   RMSE: {rmse:.2f} positions")
    
    # Show sample comparisons
    print(f"\n   Sample predictions vs actual:")
    sample = df_completed[['track', 'driver', 'grid', 'position', 'predicted_position']].head(10)
    print(sample.to_string(index=False))
    
    return df_completed

def save_predictions(df, output_path):
    """Save predictions to CSV"""
    print(f"\n💾 Saving predictions to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Sort by track and predicted position
    df_sorted = df.sort_values(['track', 'predicted_position'])
    
    # Select relevant columns
    output_cols = ['track', 'driver', 'grid', 'predicted_position', 'position', 'points']
    df_output = df_sorted[output_cols].copy()
    
    df_output.to_csv(output_path, index=False)
    print(f"✅ Saved {len(df_output)} predictions")
    
    return df_output

def main(model_type="linear"):
    print("=" * 70)
    print(f"F1 2025 Last 6 Races Prediction - {model_type.upper()} Model")
    print("=" * 70)
    
    # Load model and data
    model, scaler, model_name = load_model(model_type)
    if model is None:
        print(f"\n❌ {model_type} model not available. Please train it first.")
        print(f"   Run: python3 ml/wght_{model_type}.py")
        return
    
    df_training = load_training_data()
    df_2025 = load_2025_races()
    
    # Get last 6 races
    df_last_6, last_6_tracks, completed_races, future_races = get_last_6_races(df_2025)
    
    # Fill missing races with historical grid data
    df_last_6_filled, all_tracks, filled_tracks = fill_missing_races_with_historical(
        df_last_6, last_6_tracks, df_training
    )
    
    # Update future races list
    all_future_races = future_races + filled_tracks
    
    # Engineer features
    df_with_features = engineer_features_for_prediction(df_last_6_filled, df_training)
    
    # Make predictions
    df_predictions = make_predictions(df_with_features, model, scaler)
    
    # Evaluate on completed races
    if completed_races:
        evaluate_predictions(df_predictions, completed_races)
    
    # Save predictions with model name in filename
    output_path = OUTPUT_DIR / f"2025_races_19-24_predictions_{model_name}.csv"
    df_output = save_predictions(df_predictions, output_path)
    
    # Show predictions for future/filled races
    if all_future_races:
        print("\n🔮 Predictions for future/simulated races:")
        df_future = df_predictions[df_predictions['track'].isin(all_future_races)].sort_values(['track', 'predicted_position'])
        for track in all_future_races:
            track_data = df_future[df_future['track'] == track]
            if len(track_data) > 0:
                is_filled = track in filled_tracks
                status = "📊 Historical grid" if is_filled else "🔮 Future"
                print(f"\n   {status} - {track}:")
                track_pred = track_data[['driver', 'grid', 'predicted_position']].head(10)
                print(track_pred.to_string(index=False))
    
    print("\n" + "=" * 70)
    print("✅ Predictions complete!")
    print("=" * 70)
    print(f"\n📂 Results saved to: {output_path}")
    print(f"\n📊 Summary:")
    print(f"   Total races predicted: {len(all_tracks)}")
    print(f"   Completed races: {len(completed_races)}")
    print(f"   Future/Simulated races: {len(all_future_races)}")
    print(f"   Model used: {model_name}")

if __name__ == "__main__":
    import sys
    model_type = sys.argv[1] if len(sys.argv) > 1 else "linear"
    if model_type not in ["linear", "xgboost"]:
        print("Usage: python predict_remaining_races.py [linear|xgboost]")
        print("Defaulting to linear...")
        model_type = "linear"
    main(model_type)
