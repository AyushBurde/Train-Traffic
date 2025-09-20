"""
Unit tests for API endpoints
"""

import pytest
import requests
import json
import time
import subprocess
import os
from multiprocessing import Process

class TestAPI:
    """Test cases for API endpoints"""
    
    @pytest.fixture(scope="class")
    def api_server(self):
        """Start API server for testing"""
        # Check if models exist
        if not os.path.exists('models/best_travel_time_model.pkl'):
            pytest.skip("Models not found. Run train_models.py first.")
        
        # Start API server in background
        process = Process(target=lambda: subprocess.run(['python', 'prediction_api.py']))
        process.start()
        
        # Wait for server to start
        time.sleep(5)
        
        yield "http://localhost:5000"
        
        # Stop server
        process.terminate()
        process.join()
    
    def test_health_endpoint(self, api_server):
        """Test health check endpoint"""
        response = requests.get(f"{api_server}/health")
        assert response.status_code == 200
        
        data = response.json()
        assert 'status' in data
        assert data['status'] == 'healthy'
        assert 'models_loaded' in data
    
    def test_model_info_endpoint(self, api_server):
        """Test model info endpoint"""
        response = requests.get(f"{api_server}/model/info")
        assert response.status_code == 200
        
        data = response.json()
        assert 'travel_time_model' in data
        assert 'stop_duration_model' in data
        assert 'preprocessor_loaded' in data
    
    def test_travel_time_prediction(self, api_server):
        """Test travel time prediction endpoint"""
        test_data = {
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
        
        response = requests.post(f"{api_server}/predict/travel_time", json=test_data)
        assert response.status_code == 200
        
        data = response.json()
        assert 'travel_time_minutes' in data
        assert 'travel_time_hours' in data
        assert isinstance(data['travel_time_minutes'], (int, float))
        assert data['travel_time_minutes'] > 0
    
    def test_stop_duration_prediction(self, api_server):
        """Test stop duration prediction endpoint"""
        test_data = {
            "end_station_name": "Delhi Junction",
            "train_type": "Rajdhani",
            "priority": 5,
            "temperature": 25.0,
            "humidity": 60.0,
            "precipitation": 0.0,
            "visibility": 10.0,
            "track_condition": "Excellent"
        }
        
        response = requests.post(f"{api_server}/predict/stop_duration", json=test_data)
        assert response.status_code == 200
        
        data = response.json()
        assert 'stop_duration_minutes' in data
        assert isinstance(data['stop_duration_minutes'], (int, float))
        assert data['stop_duration_minutes'] > 0
    
    def test_complete_prediction(self, api_server):
        """Test complete prediction endpoint"""
        test_data = {
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
        
        response = requests.post(f"{api_server}/predict/complete", json=test_data)
        assert response.status_code == 200
        
        data = response.json()
        prediction = data['prediction']
        
        assert 'travel_time_minutes' in prediction
        assert 'stop_duration_minutes' in prediction
        assert 'arrival_time' in prediction
        assert 'departure_time' in prediction
        assert 'total_journey_time_minutes' in prediction
        
        # Validate time calculations
        assert prediction['total_journey_time_minutes'] == \
               prediction['travel_time_minutes'] + prediction['stop_duration_minutes']
    
    def test_batch_prediction(self, api_server):
        """Test batch prediction endpoint"""
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
                }
            ]
        }
        
        response = requests.post(f"{api_server}/predict/batch", json=batch_data)
        assert response.status_code == 200
        
        data = response.json()
        assert 'results' in data
        assert 'total_trains' in data
        assert 'successful_predictions' in data
        assert len(data['results']) == 2
        assert data['total_trains'] == 2
    
    def test_error_handling(self, api_server):
        """Test error handling for invalid requests"""
        # Test missing required fields
        response = requests.post(f"{api_server}/predict/complete", json={})
        assert response.status_code == 400
        
        data = response.json()
        assert 'error' in data
        
        # Test invalid data types
        invalid_data = {
            "start_station_name": "Mumbai Central",
            "end_station_name": "Delhi Junction",
            "route_length": "invalid",  # Should be number
            "train_type": "Rajdhani"
        }
        
        response = requests.post(f"{api_server}/predict/complete", json=invalid_data)
        # Should either return 400 or 500 depending on validation
        assert response.status_code in [400, 500]
    
    def test_response_format(self, api_server):
        """Test that all responses have consistent format"""
        test_data = {
            "start_station_name": "Mumbai Central",
            "end_station_name": "Delhi Junction",
            "route_length": 1384.0,
            "train_type": "Rajdhani"
        }
        
        # Test all prediction endpoints
        endpoints = [
            "/predict/travel_time",
            "/predict/stop_duration", 
            "/predict/complete"
        ]
        
        for endpoint in endpoints:
            response = requests.post(f"{api_server}{endpoint}", json=test_data)
            assert response.status_code == 200
            
            data = response.json()
            assert 'timestamp' in data or 'prediction' in data
            assert isinstance(data, dict)
