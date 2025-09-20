"""
Demo script to showcase the Train Arrival Time Prediction System
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import time
import json

def demo_api_predictions():
    """Demonstrate API predictions"""
    print("🚂 Train Arrival Time Prediction System Demo")
    print("=" * 50)
    
    # Check if API is running
    try:
        response = requests.get("http://localhost:5000/health", timeout=5)
        if response.status_code != 200:
            print("❌ API is not running. Please start it with: python prediction_api.py")
            return
    except:
        print("❌ API is not running. Please start it with: python prediction_api.py")
        return
    
    print("✅ API is running!")
    print()
    
    # Demo scenarios
    scenarios = [
        {
            "name": "High-Speed Rajdhani Express",
            "data": {
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
        },
        {
            "name": "Express Train in Rain",
            "data": {
                "start_station_name": "Chennai Central",
                "end_station_name": "Bangalore City",
                "route_length": 362.0,
                "train_type": "Express",
                "priority": 3,
                "current_speed": 80.0,
                "temperature": 28.0,
                "humidity": 80.0,
                "precipitation": 15.0,
                "visibility": 5.0,
                "track_condition": "Good",
                "gradient": 1.0
            }
        },
        {
            "name": "Passenger Train on Poor Track",
            "data": {
                "start_station_name": "Kolkata Howrah",
                "end_station_name": "Patna Junction",
                "route_length": 536.0,
                "train_type": "Passenger",
                "priority": 2,
                "current_speed": 60.0,
                "temperature": 30.0,
                "humidity": 75.0,
                "precipitation": 5.0,
                "visibility": 8.0,
                "track_condition": "Poor",
                "gradient": -0.5
            }
        },
        {
            "name": "Freight Train in Extreme Weather",
            "data": {
                "start_station_name": "Mumbai Central",
                "end_station_name": "Pune Junction",
                "route_length": 192.0,
                "train_type": "Freight",
                "priority": 1,
                "current_speed": 40.0,
                "temperature": 35.0,
                "humidity": 90.0,
                "precipitation": 25.0,
                "visibility": 2.0,
                "track_condition": "Fair",
                "gradient": 2.0
            }
        }
    ]
    
    print("🎯 Running Prediction Scenarios...")
    print()
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"Scenario {i}: {scenario['name']}")
        print("-" * 30)
        
        # Make prediction
        try:
            response = requests.post("http://localhost:5000/predict/complete", 
                                   json=scenario['data'], timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                prediction = result['prediction']
                
                # Display results
                print(f"📍 Route: {scenario['data']['start_station_name']} → {scenario['data']['end_station_name']}")
                print(f"🚂 Train Type: {scenario['data']['train_type']}")
                print(f"🌡️  Weather: {scenario['data']['temperature']}°C, {scenario['data']['precipitation']}mm rain")
                print(f"🛤️  Track: {scenario['data']['track_condition']}")
                print()
                
                print(f"⏱️  Travel Time: {prediction['travel_time_minutes']:.1f} minutes ({prediction['travel_time_minutes']/60:.1f} hours)")
                print(f"🛑 Stop Duration: {prediction['stop_duration_minutes']:.1f} minutes")
                print(f"🕐 Total Journey: {prediction['total_journey_time_minutes']:.1f} minutes ({prediction['total_journey_time_minutes']/60:.1f} hours)")
                
                # Parse times
                arrival_time = datetime.fromisoformat(prediction['arrival_time'])
                departure_time = datetime.fromisoformat(prediction['departure_time'])
                
                print(f"🚉 Arrival: {arrival_time.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"🚂 Departure: {departure_time.strftime('%Y-%m-%d %H:%M:%S')}")
                
            else:
                print(f"❌ Prediction failed: {response.status_code}")
                print(f"Error: {response.text}")
        
        except Exception as e:
            print(f"❌ Error: {str(e)}")
        
        print()
        time.sleep(1)  # Pause between predictions
    
    # Demo batch prediction
    print("📊 Batch Prediction Demo")
    print("-" * 30)
    
    batch_data = {
        "trains": [scenario["data"] for scenario in scenarios]
    }
    
    try:
        response = requests.post("http://localhost:5000/predict/batch", 
                               json=batch_data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"✅ Batch prediction completed!")
            print(f"📈 Successful predictions: {result['successful_predictions']}/{result['total_trains']}")
            print()
            
            # Display summary
            for i, train_result in enumerate(result['results']):
                if train_result['status'] == 'success':
                    pred = train_result['prediction']
                    print(f"Train {i+1}: {pred['travel_time_minutes']:.1f}min travel + {pred['stop_duration_minutes']:.1f}min stop = {pred['total_journey_time_minutes']:.1f}min total")
                else:
                    print(f"Train {i+1}: ❌ {train_result['error']}")
        
        else:
            print(f"❌ Batch prediction failed: {response.status_code}")
    
    except Exception as e:
        print(f"❌ Batch prediction error: {str(e)}")
    
    print()
    print("🎉 Demo completed!")
    print("💡 Try the interactive dashboard: streamlit run dashboard.py")

def demo_model_performance():
    """Demonstrate model performance metrics"""
    print("\n📊 Model Performance Analysis")
    print("=" * 50)
    
    # Load sample data
    try:
        df = pd.read_csv('data/train_data.csv')
        
        print(f"📈 Dataset Overview:")
        print(f"   Total Records: {len(df):,}")
        print(f"   Unique Trains: {df['train_id'].nunique()}")
        print(f"   Unique Routes: {df['route_id'].nunique()}")
        print(f"   Date Range: {df['timestamp'].min()[:10]} to {df['timestamp'].max()[:10]}")
        print()
        
        # Travel time statistics
        print("⏱️  Travel Time Statistics:")
        print(f"   Mean: {df['actual_travel_time'].mean():.1f} minutes")
        print(f"   Median: {df['actual_travel_time'].median():.1f} minutes")
        print(f"   Std Dev: {df['actual_travel_time'].std():.1f} minutes")
        print(f"   Min: {df['actual_travel_time'].min():.1f} minutes")
        print(f"   Max: {df['actual_travel_time'].max():.1f} minutes")
        print()
        
        # Stop duration statistics
        print("🛑 Stop Duration Statistics:")
        print(f"   Mean: {df['actual_stop_time'].mean():.1f} minutes")
        print(f"   Median: {df['actual_stop_time'].median():.1f} minutes")
        print(f"   Std Dev: {df['actual_stop_time'].std():.1f} minutes")
        print(f"   Min: {df['actual_stop_time'].min():.1f} minutes")
        print(f"   Max: {df['actual_stop_time'].max():.1f} minutes")
        print()
        
        # Performance by train type
        print("🚂 Performance by Train Type:")
        train_performance = df.groupby('train_type').agg({
            'actual_travel_time': ['mean', 'std'],
            'actual_stop_time': ['mean', 'std'],
            'delay': 'mean'
        }).round(1)
        
        for train_type in train_performance.index:
            travel_mean = train_performance.loc[train_type, ('actual_travel_time', 'mean')]
            travel_std = train_performance.loc[train_type, ('actual_travel_time', 'std')]
            stop_mean = train_performance.loc[train_type, ('actual_stop_time', 'mean')]
            stop_std = train_performance.loc[train_type, ('actual_stop_time', 'std')]
            delay_mean = train_performance.loc[train_type, ('delay', 'mean')]
            
            print(f"   {train_type:12}: Travel {travel_mean:6.1f}±{travel_std:5.1f}min, Stop {stop_mean:5.1f}±{stop_std:4.1f}min, Delay {delay_mean:5.1f}min")
        
        print()
        
        # Weather impact
        print("🌤️  Weather Impact Analysis:")
        df['weather_severity'] = (
            (df['precipitation'] > 10).astype(int) * 3 +
            (df['visibility'] < 2).astype(int) * 2 +
            (df['wind_speed'] > 20).astype(int) * 1 +
            ((df['temperature'] > 35) | (df['temperature'] < 5)).astype(int) * 1
        )
        
        weather_impact = df.groupby('weather_severity').agg({
            'actual_travel_time': 'mean',
            'actual_stop_time': 'mean',
            'delay': 'mean'
        }).round(1)
        
        weather_labels = {0: 'Clear', 1: 'Mild', 2: 'Moderate', 3: 'Severe', 4: 'Extreme'}
        
        for severity, label in weather_labels.items():
            if severity in weather_impact.index:
                travel_time = weather_impact.loc[severity, 'actual_travel_time']
                stop_time = weather_impact.loc[severity, 'actual_stop_time']
                delay = weather_impact.loc[severity, 'delay']
                
                print(f"   {label:10}: Travel {travel_time:6.1f}min, Stop {stop_time:5.1f}min, Delay {delay:5.1f}min")
        
    except FileNotFoundError:
        print("❌ Dataset not found. Please run: python data_generator.py")
    except Exception as e:
        print(f"❌ Error loading dataset: {str(e)}")

def main():
    """Main demo function"""
    print("🚂 Train Arrival Time Prediction System")
    print("Comprehensive Demo and Analysis")
    print("=" * 60)
    print()
    
    # Check if models exist
    import os
    if not os.path.exists('models/best_travel_time_model.pkl'):
        print("❌ Models not found. Please run: python train_models.py")
        return
    
    # Run demos
    demo_model_performance()
    demo_api_predictions()
    
    print("\n🎯 Next Steps:")
    print("1. Start the API: python prediction_api.py")
    print("2. Launch dashboard: streamlit run dashboard.py")
    print("3. Explore the interactive features")
    print("4. Integrate with your railway systems")

if __name__ == "__main__":
    main()
