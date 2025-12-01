"""
Test script for F1 Prediction API
"""

import requests
import json

BASE_URL = "http://127.0.0.1:8000"

def test_root():
    print("=" * 60)
    print("Testing Root Endpoint")
    print("=" * 60)
    response = requests.get(f"{BASE_URL}/")
    print(json.dumps(response.json(), indent=2))
    print()

def test_models_info():
    print("=" * 60)
    print("Testing Models Info")
    print("=" * 60)
    response = requests.get(f"{BASE_URL}/models")
    print(json.dumps(response.json(), indent=2))
    print()

def test_single_prediction():
    print("=" * 60)
    print("Testing Single Driver Prediction")
    print("=" * 60)
    
    payload = {
        "driver": "Max Verstappen",
        "grid": 1.0,
        "track": "Monaco"
    }
    
    print(f"Request: {json.dumps(payload, indent=2)}")
    response = requests.post(f"{BASE_URL}/predict/linear", json=payload)
    print(f"\nResponse:")
    print(json.dumps(response.json(), indent=2))
    print()

def test_race_prediction():
    print("=" * 60)
    print("Testing Full Race Prediction")
    print("=" * 60)
    
    payload = {
        "track": "Monaco",
        "grid_positions": {
            "Max Verstappen": 1,
            "Charles Leclerc": 2,
            "Lando Norris": 3,
            "Oscar Piastri": 4,
            "Carlos Sainz": 5
        }
    }
    
    print(f"Request: {json.dumps(payload, indent=2)}")
    response = requests.post(f"{BASE_URL}/predict-race/linear", json=payload)
    result = response.json()
    
    print(f"\nPredicted Race Results for {result['track']}:")
    print(f"{'Position':<10} {'Driver':<20} {'Grid':<10} {'Predicted':<15}")
    print("-" * 55)
    
    for i, pred in enumerate(result['predictions'], 1):
        print(f"{i:<10} {pred['driver']:<20} {pred['grid']:<10.0f} {pred['predicted_position']:<15.2f}")
    print()

if __name__ == "__main__":
    try:
        print("\n🏎️  F1 Prediction API Test Suite\n")
        
        test_root()
        test_models_info()
        test_single_prediction()
        test_race_prediction()
        
        print("=" * 60)
        print("✅ All tests completed!")
        print("=" * 60)
        
    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to API")
        print("Make sure the server is running:")
        print("  uvicorn app.main:app --port 8000")
    except Exception as e:
        print(f"❌ Error: {e}")
