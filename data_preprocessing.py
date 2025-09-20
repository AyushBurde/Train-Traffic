"""
Data Preprocessing and Feature Engineering for Train Arrival Time Prediction
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression
import joblib
from config import DATA_CONFIG, MODEL_CONFIG, FEATURE_CONFIG

class DataPreprocessor:
    def __init__(self):
        self.label_encoders = {}
        self.scalers = {}
        self.feature_selector = None
        self.selected_features = None
        
    def load_data(self, filepath):
        """Load data from CSV file"""
        df = pd.read_csv(filepath)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['arrival_time'] = pd.to_datetime(df['arrival_time'])
        return df
    
    def create_time_features(self, df):
        """Create time-based features"""
        df = df.copy()
        
        # Extract time features
        df['hour_of_day'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        df['day_of_year'] = df['timestamp'].dt.dayofyear
        df['is_weekend'] = df['timestamp'].dt.dayofweek >= 5
        df['is_holiday'] = df['is_holiday'].astype(int)
        
        # Cyclical encoding for time features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        return df
    
    def create_weather_features(self, df):
        """Create weather-based features"""
        df = df.copy()
        
        # Weather severity indicators
        df['heavy_rain'] = (df['precipitation'] > 10).astype(int)
        df['low_visibility'] = (df['visibility'] < 2).astype(int)
        df['high_wind'] = (df['wind_speed'] > 20).astype(int)
        df['extreme_temp'] = ((df['temperature'] > 35) | (df['temperature'] < 5)).astype(int)
        
        # Weather interaction features
        df['weather_severity'] = (
            df['heavy_rain'] * 3 +
            df['low_visibility'] * 2 +
            df['high_wind'] * 1 +
            df['extreme_temp'] * 1
        )
        
        return df
    
    def create_train_features(self, df):
        """Create train-specific features"""
        df = df.copy()
        
        # Train type encoding
        train_type_priority = {
            'Rajdhani': 5,
            'Shatabdi': 5,
            'Superfast': 4,
            'Express': 3,
            'Mail': 2,
            'Passenger': 1,
            'Freight': 1
        }
        
        df['train_priority_score'] = df['train_type'].map(train_type_priority)
        
        # Speed efficiency
        df['speed_efficiency'] = df['current_speed'] / df['route_length'] * 100
        
        # Delay history features
        df['has_previous_delay'] = (df['previous_delay'] > 0).astype(int)
        df['delay_severity'] = pd.cut(df['previous_delay'], 
                                    bins=[0, 5, 15, 30, float('inf')], 
                                    labels=[0, 1, 2, 3]).astype(int)
        
        return df
    
    def create_station_features(self, df):
        """Create station-specific features"""
        df = df.copy()
        
        # Station capacity utilization
        df['start_capacity_util'] = df['start_station_congestion'] * df['start_platform_count']
        df['end_capacity_util'] = df['end_station_congestion'] * df['end_platform_count']
        
        # Station type priority
        station_type_priority = {
            'Terminal': 4,
            'Junction': 3,
            'Major': 2,
            'Minor': 1
        }
        
        df['start_station_priority'] = df['start_station_type'].map(station_type_priority)
        df['end_station_priority'] = df['end_station_type'].map(station_type_priority)
        
        return df
    
    def create_route_features(self, df):
        """Create route-specific features"""
        df = df.copy()
        
        # Route complexity
        df['route_complexity'] = (
            df['junction_count'] * 0.3 +
            df['signal_density'] * 0.2 +
            abs(df['gradient']) * 0.1 +
            (df['track_condition'] == 'Poor').astype(int) * 0.4
        )
        
        # Track condition encoding
        track_condition_map = {
            'Excellent': 4,
            'Good': 3,
            'Fair': 2,
            'Poor': 1
        }
        df['track_condition_score'] = df['track_condition'].map(track_condition_map)
        
        # Distance-based features
        df['distance_per_junction'] = df['route_length'] / (df['junction_count'] + 1)
        df['signals_per_km'] = df['signal_density']
        
        return df
    
    def create_interaction_features(self, df):
        """Create interaction features between different categories"""
        df = df.copy()
        
        # Weather and route interactions
        df['weather_route_impact'] = df['weather_severity'] * df['route_complexity']
        
        # Train and station interactions
        df['train_station_priority'] = df['train_priority_score'] * df['end_station_priority']
        
        # Time and weather interactions
        df['rush_hour_weather'] = (
            ((df['hour_of_day'].between(7, 9)) | (df['hour_of_day'].between(17, 19))) *
            df['weather_severity']
        )
        
        # Congestion and delay interactions
        df['congestion_delay_risk'] = (
            df['end_station_congestion'] * 
            df['has_previous_delay'] * 
            df['route_complexity']
        )
        
        return df
    
    def encode_categorical_features(self, df, fit=True):
        """Encode categorical features"""
        df = df.copy()
        
        categorical_columns = [
            'train_type', 'track_condition', 'start_station_type', 'end_station_type'
        ]
        
        for col in categorical_columns:
            if fit:
                le = LabelEncoder()
                df[col + '_encoded'] = le.fit_transform(df[col])
                self.label_encoders[col] = le
            else:
                if col in self.label_encoders:
                    # Handle unseen categories
                    le = self.label_encoders[col]
                    df[col + '_encoded'] = df[col].map(
                        dict(zip(le.classes_, le.transform(le.classes_)))
                    ).fillna(-1)  # -1 for unseen categories
        
        return df
    
    def scale_features(self, df, fit=True):
        """Scale numerical features"""
        df = df.copy()
        
        # Features to scale
        scale_columns = [
            'route_length', 'gradient', 'signal_density', 'junction_count',
            'start_platform_count', 'start_station_congestion',
            'end_platform_count', 'end_station_congestion',
            'temperature', 'humidity', 'precipitation', 'wind_speed', 'visibility',
            'current_speed', 'previous_delay', 'base_travel_time', 'base_stop_time'
        ]
        
        # Add new features to scale
        new_features = [
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos',
            'weather_severity', 'train_priority_score', 'speed_efficiency',
            'start_capacity_util', 'end_capacity_util', 'start_station_priority',
            'end_station_priority', 'route_complexity', 'track_condition_score',
            'distance_per_junction', 'signals_per_km', 'weather_route_impact',
            'train_station_priority', 'rush_hour_weather', 'congestion_delay_risk'
        ]
        
        scale_columns.extend(new_features)
        
        # Only scale columns that exist in the dataframe
        scale_columns = [col for col in scale_columns if col in df.columns]
        
        if fit:
            scaler = StandardScaler()
            df[scale_columns] = scaler.fit_transform(df[scale_columns])
            self.scalers['standard'] = scaler
        else:
            if 'standard' in self.scalers:
                scaler = self.scalers['standard']
                df[scale_columns] = scaler.transform(df[scale_columns])
        
        return df
    
    def select_features(self, X, y, k=50):
        """Select top k features using statistical tests"""
        if self.feature_selector is None:
            self.feature_selector = SelectKBest(score_func=f_regression, k=k)
            X_selected = self.feature_selector.fit_transform(X, y)
            self.selected_features = X.columns[self.feature_selector.get_support()].tolist()
        else:
            X_selected = self.feature_selector.transform(X)
        
        return X_selected
    
    def prepare_features(self, df, target_columns=['actual_travel_time', 'actual_stop_time']):
        """Prepare all features for ML models"""
        print("Creating time features...")
        df = self.create_time_features(df)
        
        print("Creating weather features...")
        df = self.create_weather_features(df)
        
        print("Creating train features...")
        df = self.create_train_features(df)
        
        print("Creating station features...")
        df = self.create_station_features(df)
        
        print("Creating route features...")
        df = self.create_route_features(df)
        
        print("Creating interaction features...")
        df = self.create_interaction_features(df)
        
        print("Encoding categorical features...")
        df = self.encode_categorical_features(df, fit=True)
        
        print("Scaling features...")
        df = self.scale_features(df, fit=True)
        
        # Prepare feature matrix
        feature_columns = [col for col in df.columns if col not in [
            'record_id', 'timestamp', 'train_id', 'start_station_name', 'end_station_name',
            'route_id', 'start_station_id', 'end_station_id', 'arrival_time',
            'actual_travel_time', 'actual_stop_time', 'delay', 'base_travel_time', 'base_stop_time',
            'train_type', 'track_condition', 'start_station_type', 'end_station_type'
        ]]
        
        X = df[feature_columns]
        
        # Handle missing values
        X = X.fillna(0)
        
        return X, df[target_columns]
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into train and test sets"""
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    def save_preprocessor(self, filepath):
        """Save preprocessor objects"""
        preprocessor_data = {
            'label_encoders': self.label_encoders,
            'scalers': self.scalers,
            'feature_selector': self.feature_selector,
            'selected_features': self.selected_features
        }
        joblib.dump(preprocessor_data, filepath)
        print(f"Preprocessor saved to {filepath}")
    
    def load_preprocessor(self, filepath):
        """Load preprocessor objects"""
        preprocessor_data = joblib.load(filepath)
        self.label_encoders = preprocessor_data['label_encoders']
        self.scalers = preprocessor_data['scalers']
        self.feature_selector = preprocessor_data['feature_selector']
        self.selected_features = preprocessor_data['selected_features']
        print(f"Preprocessor loaded from {filepath}")

def main():
    """Test the data preprocessing pipeline"""
    print("Testing data preprocessing pipeline...")
    
    # Load data
    preprocessor = DataPreprocessor()
    df = preprocessor.load_data('data/train_data.csv')
    
    print(f"Original data shape: {df.shape}")
    print(f"Original columns: {len(df.columns)}")
    
    # Prepare features
    X, y = preprocessor.prepare_features(df)
    
    print(f"Processed data shape: {X.shape}")
    print(f"Feature columns: {len(X.columns)}")
    print(f"Target columns: {list(y.columns)}")
    
    # Split data
    X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)
    
    print(f"Train set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    
    # Save preprocessor
    preprocessor.save_preprocessor('models/preprocessor.pkl')
    
    print("Data preprocessing completed successfully!")

if __name__ == "__main__":
    main()
