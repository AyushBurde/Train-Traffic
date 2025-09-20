"""
Configuration file for the Train Arrival Time Prediction System
"""

import os
from datetime import datetime, timedelta

# Data Configuration
DATA_CONFIG = {
    'dataset_path': 'data/',
    'train_data_file': 'train_data.csv',
    'test_data_file': 'test_data.csv',
    'model_save_path': 'models/',
    'results_path': 'results/'
}

# Model Configuration
MODEL_CONFIG = {
    'test_size': 0.2,
    'random_state': 42,
    'cv_folds': 5,
    'models_to_train': [
        'linear_regression',
        'random_forest',
        'xgboost',
        'lightgbm',
        'catboost',
        'neural_network'
    ]
}

# Feature Engineering Configuration
FEATURE_CONFIG = {
    'time_features': [
        'hour_of_day',
        'day_of_week',
        'month',
        'is_weekend',
        'is_holiday'
    ],
    'weather_features': [
        'temperature',
        'humidity',
        'precipitation',
        'wind_speed',
        'visibility'
    ],
    'train_features': [
        'train_type',
        'priority',
        'capacity',
        'current_speed',
        'previous_delay'
    ],
    'station_features': [
        'station_type',
        'platform_count',
        'station_congestion',
        'distance_from_origin'
    ],
    'route_features': [
        'route_length',
        'track_condition',
        'gradient',
        'signal_density',
        'junction_count'
    ]
}

# API Configuration
API_CONFIG = {
    'host': '0.0.0.0',
    'port': 5000,
    'debug': True
}

# Create necessary directories
for path in DATA_CONFIG.values():
    if not os.path.exists(path):
        os.makedirs(path)
