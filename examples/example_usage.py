"""
Example usage of Train Arrival Time Prediction System
"""

import requests
import json
from datetime import datetime

def example_single_prediction():
    """Example of single train prediction"""
    print("Single Train Prediction Example")
    print("=" * 40)
    
    # API endpoint
    api_url = "http://localhost:5000"
    
    # Sample data
    train_data = {
        "start_station_name": "Mumbai Central",
        "end_station_name": "Delhi Junction",
        "route_length": 1384.0,
        "train_type": "Rajdhani",
        "priority": 5,
        "current_speed": 120.0,
        "temperature": 25.0,
        "humidity": 60.0,
        "precipitation": 0.0,
        "visibility": 10.0,
        "track_condition": "Excellent",
        "gradient": 0.0
    }
    
    try:
        # Make prediction
        response = requests.post(f"{api_url}/predict/complete", json=train_data)
        
        if response.status_code == 200:
            result = response.json()
            prediction = result['prediction']
            
            print(f"Route: {train_data['start_station_name']} -> {train_data['end_station_name']}")
            print(f"Train Type: {train_data['train_type']}")
            print(f"Travel Time: {prediction['travel_time_minutes']:.1f} minutes")
            print(f"Stop Duration: {prediction['stop_duration_minutes']:.1f} minutes")
            print(f"Arrival Time: {prediction['arrival_time']}")
            print(f"Departure Time: {prediction['departure_time']}")
        else:
            print(f"Error: {response.status_code} - {response.text}")
    
    except requests.exceptions.ConnectionError:
        print("API server is not running. Please start it with: python prediction_api.py")
    except Exception as e:
        print(f"Error: {str(e)}")

def example_batch_prediction():
    """Example of batch prediction"""
    print("\nBatch Prediction Example")
    print("=" * 40)
    
    api_url = "http://localhost:5000"
    
    # Multiple trains
    batch_data = {
        "trains": [
            {
                "start_station_name": "Mumbai Central",
                "end_station_name": "Delhi Junction",
                "route_length": 1384.0,
                "train_type": "Rajdhani"
            },
            {
                "start_station_name": "Chennai Central",
                "end_station_name": "Bangalore City",
                "route_length": 362.0,
                "train_type": "Express"
            },
            {
                "start_station_name": "Kolkata Howrah",
                "end_station_name": "Patna Junction",
                "route_length": 536.0,
                "train_type": "Passenger"
            }
        ]
    }
    
    try:
        response = requests.post(f"{api_url}/predict/batch", json=batch_data)
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"Batch prediction completed!")
            print(f"Successful predictions: {result['successful_predictions']}/{result['total_trains']}")
            print()
            
            for i, train_result in enumerate(result['results']):
                if train_result['status'] == 'success':
                    pred = train_result['prediction']
                    print(f"Train {i+1}: {pred['travel_time_minutes']:.1f}min travel + {pred['stop_duration_minutes']:.1f}min stop")
                else:
                    print(f"Train {i+1}: Error - {train_result['error']}")
        else:
            print(f"Error: {response.status_code} - {response.text}")
    
    except requests.exceptions.ConnectionError:
        print("API server is not running. Please start it with: python prediction_api.py")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    print("Train Arrival Time Prediction System - Example Usage")
    print("=" * 60)
    
    # Check if API is running
    try:
        response = requests.get("http://localhost:5000/health", timeout=5)
        if response.status_code == 200:
            print("API is running!")
            example_single_prediction()
            example_batch_prediction()
        else:
            print("API is not responding properly")
    except:
        print("API is not running. Please start it with: python prediction_api.py")
        print("\nTo run this example:")
        print("1. Start API: python prediction_api.py")
        print("2. Run example: python examples/example_usage.py")
