"""
Real-time Prediction API for Train Arrival Time and Stop Duration
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
import os
from data_preprocessing import DataPreprocessor

app = Flask(__name__)
CORS(app)

class TrainPredictionAPI:
    def __init__(self):
        self.preprocessor = None
        self.travel_time_model = None
        self.stop_duration_model = None
        self.load_models()
    
    def load_models(self):
        """Load trained models and preprocessor"""
        try:
            # Load preprocessor
            self.preprocessor = DataPreprocessor()
            self.preprocessor.load_preprocessor('models/preprocessor.pkl')
            
            # Load models
            self.travel_time_model = joblib.load('models/best_travel_time_model.pkl')
            self.stop_duration_model = joblib.load('models/best_stop_duration_model.pkl')
            
            print("Models loaded successfully!")
        except Exception as e:
            print(f"Error loading models: {str(e)}")
    
    def prepare_prediction_data(self, input_data):
        """Prepare input data for prediction"""
        # Create a DataFrame from input data
        df = pd.DataFrame([input_data])
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Add required columns with default values if not provided
        default_values = {
            'train_id': 'T0001',
            'priority': 3,
            'route_id': 'RT001',
            'track_condition': 'Good',
            'gradient': 0.0,
            'signal_density': 1.0,
            'junction_count': 2,
            'start_station_type': 'Major',
            'start_platform_count': 4,
            'start_station_congestion': 0.5,
            'end_station_type': 'Major',
            'end_platform_count': 4,
            'end_station_congestion': 0.5,
            'temperature': 25.0,
            'humidity': 60.0,
            'precipitation': 0.0,
            'wind_speed': 10.0,
            'visibility': 10.0,
            'current_speed': 80.0,
            'previous_delay': 0.0,
            'base_travel_time': 120.0,
            'base_stop_time': 5.0
        }
        
        for key, value in default_values.items():
            if key not in df.columns:
                df[key] = value
        
        # Process the data using the preprocessor
        X, _ = self.preprocessor.prepare_features(df)
        
        return X
    
    def predict_travel_time(self, input_data):
        """Predict travel time from station A to station B"""
        try:
            X = self.prepare_prediction_data(input_data)
            prediction = self.travel_time_model.predict(X)[0]
            return float(prediction)
        except Exception as e:
            print(f"Error predicting travel time: {str(e)}")
            return None
    
    def predict_stop_duration(self, input_data):
        """Predict stop duration at station B"""
        try:
            X = self.prepare_prediction_data(input_data)
            prediction = self.stop_duration_model.predict(X)[0]
            return float(prediction)
        except Exception as e:
            print(f"Error predicting stop duration: {str(e)}")
            return None
    
    def predict_arrival_time(self, input_data):
        """Predict arrival time and stop duration"""
        try:
            # Get current timestamp if not provided
            if 'timestamp' not in input_data:
                input_data['timestamp'] = datetime.now().isoformat()
            
            # Predict travel time
            travel_time_minutes = self.predict_travel_time(input_data)
            if travel_time_minutes is None:
                return None
            
            # Predict stop duration
            stop_duration_minutes = self.predict_stop_duration(input_data)
            if stop_duration_minutes is None:
                return None
            
            # Calculate arrival time
            start_time = pd.to_datetime(input_data['timestamp'])
            arrival_time = start_time + timedelta(minutes=travel_time_minutes)
            
            # Calculate departure time (arrival + stop duration)
            departure_time = arrival_time + timedelta(minutes=stop_duration_minutes)
            
            return {
                'travel_time_minutes': travel_time_minutes,
                'stop_duration_minutes': stop_duration_minutes,
                'arrival_time': arrival_time.isoformat(),
                'departure_time': departure_time.isoformat(),
                'total_journey_time_minutes': travel_time_minutes + stop_duration_minutes
            }
        except Exception as e:
            print(f"Error predicting arrival time: {str(e)}")
            return None

# Initialize the API
api = TrainPredictionAPI()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'models_loaded': api.travel_time_model is not None and api.stop_duration_model is not None
    })

@app.route('/predict/travel_time', methods=['POST'])
def predict_travel_time():
    """Predict travel time between stations"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Required fields
        required_fields = ['start_station_name', 'end_station_name', 'route_length']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        prediction = api.predict_travel_time(data)
        
        if prediction is None:
            return jsonify({'error': 'Failed to make prediction'}), 500
        
        return jsonify({
            'travel_time_minutes': prediction,
            'travel_time_hours': round(prediction / 60, 2),
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict/stop_duration', methods=['POST'])
def predict_stop_duration():
    """Predict stop duration at destination station"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Required fields
        required_fields = ['end_station_name', 'train_type']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        prediction = api.predict_stop_duration(data)
        
        if prediction is None:
            return jsonify({'error': 'Failed to make prediction'}), 500
        
        return jsonify({
            'stop_duration_minutes': prediction,
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict/complete', methods=['POST'])
def predict_complete():
    """Predict both travel time and stop duration"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Required fields
        required_fields = ['start_station_name', 'end_station_name', 'route_length', 'train_type']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        prediction = api.predict_arrival_time(data)
        
        if prediction is None:
            return jsonify({'error': 'Failed to make prediction'}), 500
        
        return jsonify({
            'prediction': prediction,
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """Predict for multiple trains"""
    try:
        data = request.get_json()
        
        if not data or 'trains' not in data:
            return jsonify({'error': 'No trains data provided'}), 400
        
        results = []
        for i, train_data in enumerate(data['trains']):
            try:
                prediction = api.predict_arrival_time(train_data)
                if prediction:
                    results.append({
                        'train_index': i,
                        'prediction': prediction,
                        'status': 'success'
                    })
                else:
                    results.append({
                        'train_index': i,
                        'error': 'Failed to make prediction',
                        'status': 'error'
                    })
            except Exception as e:
                results.append({
                    'train_index': i,
                    'error': str(e),
                    'status': 'error'
                })
        
        return jsonify({
            'results': results,
            'total_trains': len(data['trains']),
            'successful_predictions': len([r for r in results if r['status'] == 'success']),
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/model/info', methods=['GET'])
def model_info():
    """Get information about loaded models"""
    return jsonify({
        'travel_time_model': {
            'type': type(api.travel_time_model).__name__ if api.travel_time_model else None,
            'loaded': api.travel_time_model is not None
        },
        'stop_duration_model': {
            'type': type(api.stop_duration_model).__name__ if api.stop_duration_model else None,
            'loaded': api.stop_duration_model is not None
        },
        'preprocessor_loaded': api.preprocessor is not None,
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    print("Starting Train Prediction API...")
    print("API Endpoints:")
    print("  GET  /health - Health check")
    print("  POST /predict/travel_time - Predict travel time")
    print("  POST /predict/stop_duration - Predict stop duration")
    print("  POST /predict/complete - Predict both travel time and stop duration")
    print("  POST /predict/batch - Predict for multiple trains")
    print("  GET  /model/info - Get model information")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
