"""
Unit tests for ML models
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os
from data_preprocessing import DataPreprocessor
from train_models import SimpleTrainPredictor

class TestMLModels:
    """Test cases for ML models"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        return pd.DataFrame({
            'start_station_name': ['Mumbai Central', 'Delhi Junction'],
            'end_station_name': ['Delhi Junction', 'Chennai Central'],
            'route_length': [1384.0, 2200.0],
            'train_type': ['Rajdhani', 'Express'],
            'priority': [5, 3],
            'current_speed': [120.0, 80.0],
            'temperature': [25.0, 28.0],
            'humidity': [60.0, 70.0],
            'precipitation': [0.0, 5.0],
            'visibility': [10.0, 8.0],
            'track_condition': ['Excellent', 'Good'],
            'gradient': [0.0, 1.0],
            'timestamp': [pd.Timestamp.now(), pd.Timestamp.now()]
        })
    
    def test_model_files_exist(self):
        """Test that model files exist"""
        assert os.path.exists('models/best_travel_time_model.pkl')
        assert os.path.exists('models/best_stop_duration_model.pkl')
        assert os.path.exists('models/preprocessor.pkl')
    
    def test_model_loading(self):
        """Test that models can be loaded"""
        travel_model = joblib.load('models/best_travel_time_model.pkl')
        stop_model = joblib.load('models/best_stop_duration_model.pkl')
        
        assert travel_model is not None
        assert stop_model is not None
    
    def test_preprocessor_loading(self):
        """Test that preprocessor can be loaded"""
        preprocessor = DataPreprocessor()
        preprocessor.load_preprocessor('models/preprocessor.pkl')
        
        assert preprocessor is not None
        assert hasattr(preprocessor, 'label_encoders')
        assert hasattr(preprocessor, 'scalers')
    
    def test_prediction_accuracy(self):
        """Test prediction accuracy on sample data"""
        # Load models
        travel_model = joblib.load('models/best_travel_time_model.pkl')
        stop_model = joblib.load('models/best_stop_duration_model.pkl')
        preprocessor = DataPreprocessor()
        preprocessor.load_preprocessor('models/preprocessor.pkl')
        
        # Load test data
        df = pd.read_csv('data/train_data.csv')
        X, y = preprocessor.prepare_features(df)
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Test travel time model
        y_travel_pred = travel_model.predict(X_test)
        travel_mae = mean_absolute_error(y_test['actual_travel_time'], y_travel_pred)
        travel_r2 = r2_score(y_test['actual_travel_time'], y_travel_pred)
        
        # Test stop duration model
        y_stop_pred = stop_model.predict(X_test)
        stop_mae = mean_absolute_error(y_test['actual_stop_time'], y_stop_pred)
        stop_r2 = r2_score(y_test['actual_stop_time'], y_stop_pred)
        
        # Assertions
        assert travel_mae < 200  # MAE should be reasonable
        assert travel_r2 > 0.8   # R² should be good
        assert stop_mae < 10     # Stop duration MAE should be small
        assert stop_r2 > 0.8     # R² should be good
    
    def test_prediction_consistency(self):
        """Test that predictions are consistent"""
        travel_model = joblib.load('models/best_travel_time_model.pkl')
        stop_model = joblib.load('models/best_stop_duration_model.pkl')
        preprocessor = DataPreprocessor()
        preprocessor.load_preprocessor('models/preprocessor.pkl')
        
        # Create test data
        test_data = {
            'start_station_name': 'Mumbai Central',
            'end_station_name': 'Delhi Junction',
            'route_length': 1384.0,
            'train_type': 'Rajdhani',
            'priority': 5,
            'current_speed': 120.0,
            'temperature': 25.0,
            'humidity': 60.0,
            'precipitation': 0.0,
            'visibility': 10.0,
            'track_condition': 'Excellent',
            'gradient': 0.0,
            'timestamp': pd.Timestamp.now()
        }
        
        # Prepare data
        X = preprocessor.prepare_prediction_data(pd.DataFrame([test_data]))
        
        # Make predictions
        travel_time_1 = travel_model.predict(X)[0]
        stop_duration_1 = stop_model.predict(X)[0]
        
        travel_time_2 = travel_model.predict(X)[0]
        stop_duration_2 = stop_model.predict(X)[0]
        
        # Predictions should be identical
        assert travel_time_1 == travel_time_2
        assert stop_duration_1 == stop_duration_2
    
    def test_prediction_range(self):
        """Test that predictions are in reasonable ranges"""
        travel_model = joblib.load('models/best_travel_time_model.pkl')
        stop_model = joblib.load('models/best_stop_duration_model.pkl')
        preprocessor = DataPreprocessor()
        preprocessor.load_preprocessor('models/preprocessor.pkl')
        
        # Test with different scenarios
        test_scenarios = [
            {
                'name': 'Short route',
                'data': {
                    'start_station_name': 'Mumbai Central',
                    'end_station_name': 'Pune Junction',
                    'route_length': 192.0,
                    'train_type': 'Express',
                    'priority': 3,
                    'current_speed': 80.0,
                    'temperature': 25.0,
                    'humidity': 60.0,
                    'precipitation': 0.0,
                    'visibility': 10.0,
                    'track_condition': 'Good',
                    'gradient': 0.0,
                    'timestamp': pd.Timestamp.now()
                },
                'expected_travel_range': (60, 300),  # 1-5 hours
                'expected_stop_range': (1, 30)       # 1-30 minutes
            },
            {
                'name': 'Long route',
                'data': {
                    'start_station_name': 'Mumbai Central',
                    'end_station_name': 'Delhi Junction',
                    'route_length': 1384.0,
                    'train_type': 'Rajdhani',
                    'priority': 5,
                    'current_speed': 120.0,
                    'temperature': 25.0,
                    'humidity': 60.0,
                    'precipitation': 0.0,
                    'visibility': 10.0,
                    'track_condition': 'Excellent',
                    'gradient': 0.0,
                    'timestamp': pd.Timestamp.now()
                },
                'expected_travel_range': (600, 1200),  # 10-20 hours
                'expected_stop_range': (2, 15)         # 2-15 minutes
            }
        ]
        
        for scenario in test_scenarios:
            X = preprocessor.prepare_prediction_data(pd.DataFrame([scenario['data']]))
            
            travel_time = travel_model.predict(X)[0]
            stop_duration = stop_model.predict(X)[0]
            
            # Check ranges
            assert scenario['expected_travel_range'][0] <= travel_time <= scenario['expected_travel_range'][1], \
                f"Travel time {travel_time} not in expected range for {scenario['name']}"
            
            assert scenario['expected_stop_range'][0] <= stop_duration <= scenario['expected_stop_range'][1], \
                f"Stop duration {stop_duration} not in expected range for {scenario['name']}"
