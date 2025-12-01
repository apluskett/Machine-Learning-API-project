"""
Compare Linear Regression vs XGBoost model performance
"""

import pandas as pd
import numpy as np

print("=" * 70)
print("F1 Model Comparison: Linear Regression vs XGBoost")
print("=" * 70)

# Model Performance Summary
print("\n📊 Model Performance Comparison")
print("-" * 70)

models = {
    "Linear Regression": {
        "Training R²": 0.6313,
        "Validation R²": 0.6114,
        "Validation MAE": 2.47,
        "Validation RMSE": 3.34
    },
    "XGBoost": {
        "Training R²": 0.8842,
        "Validation R²": 0.5783,
        "Validation MAE": 2.60,
        "Validation RMSE": 3.48
    }
}

# Print comparison table
print(f"\n{'Metric':<20} {'Linear Regression':<20} {'XGBoost':<20} {'Winner':<15}")
print("-" * 75)

metrics = ["Training R²", "Validation R²", "Validation MAE", "Validation RMSE"]

for metric in metrics:
    lin_val = models["Linear Regression"][metric]
    xgb_val = models["XGBoost"][metric]
    
    # Determine winner (higher is better for R², lower is better for MAE/RMSE)
    if "R²" in metric:
        winner = "Linear" if lin_val > xgb_val else "XGBoost"
        better = "↑"
    else:
        winner = "Linear" if lin_val < xgb_val else "XGBoost"
        better = "↓"
    
    print(f"{metric:<20} {lin_val:<20.4f} {xgb_val:<20.4f} {winner} {better}")

# Real-world performance on completed 2025 races
print("\n\n📈 Real-World Performance (2025 Races 19-22)")
print("-" * 70)

real_world = {
    "Linear Regression": {
        "MAE": 2.21,
        "RMSE": 3.01
    },
    "XGBoost": {
        "MAE": 2.49,
        "RMSE": 3.18
    }
}

print(f"\n{'Metric':<20} {'Linear Regression':<20} {'XGBoost':<20} {'Winner':<15}")
print("-" * 75)

for metric in ["MAE", "RMSE"]:
    lin_val = real_world["Linear Regression"][metric]
    xgb_val = real_world["XGBoost"][metric]
    winner = "Linear" if lin_val < xgb_val else "XGBoost"
    
    print(f"{metric:<20} {lin_val:<20.2f} {xgb_val:<20.2f} {winner} ✓")

# Analysis
print("\n\n🔍 Analysis")
print("-" * 70)

print("\n✅ Linear Regression Strengths:")
print("   - Better generalization (higher validation R²: 0.6114 vs 0.5783)")
print("   - Lower real-world error (MAE: 2.21 vs 2.49)")
print("   - Less overfitting (gap between train/val R²: 0.02 vs 0.31)")
print("   - More stable predictions")
print("   - Simpler, more interpretable")

print("\n✅ XGBoost Strengths:")
print("   - Much better training fit (R²: 0.8842 vs 0.6313)")
print("   - Captures non-linear patterns")
print("   - Feature importance ranking")
print("   - Could improve with hyperparameter tuning")

print("\n\n🏆 Recommendation:")
print("-" * 70)
print("\n   For this F1 prediction task: **Linear Regression** performs better!")
print("\n   Reasons:")
print("   • Less overfitting on validation data")
print("   • Better real-world performance (2.21 vs 2.49 MAE)")
print("   • F1 has high inherent randomness - simpler model generalizes better")
print("   • XGBoost may be memorizing training patterns that don't transfer")

print("\n   Next steps to improve XGBoost:")
print("   • Reduce max_depth (try 4 or 5)")
print("   • Increase regularization (reg_alpha, reg_lambda)")
print("   • Reduce n_estimators")
print("   • Add more diverse features")
print("   • Cross-validation for hyperparameter tuning")

print("\n" + "=" * 70)
print("Both models are available in the API!")
print("  Linear:  curl -X POST http://localhost:8000/predict/linear")
print("  XGBoost: curl -X POST http://localhost:8000/predict/xgboost")
print("=" * 70)
