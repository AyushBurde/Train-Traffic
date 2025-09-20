"""
Synthetic Railway Dataset Generator
Creates realistic train data for ML model training
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from config import DATA_CONFIG, FEATURE_CONFIG

class RailwayDataGenerator:
    def __init__(self, num_records=10000):
        self.num_records = num_records
        self.stations = self._generate_stations()
        self.routes = self._generate_routes()
        self.train_types = ['Express', 'Mail', 'Passenger', 'Freight', 'Superfast', 'Rajdhani', 'Shatabdi']
        
    def _generate_stations(self):
        """Generate realistic station data"""
        stations = []
        station_names = [
            'Mumbai Central', 'Delhi Junction', 'Chennai Central', 'Kolkata Howrah',
            'Bangalore City', 'Hyderabad Deccan', 'Pune Junction', 'Ahmedabad',
            'Jaipur Junction', 'Lucknow Charbagh', 'Patna Junction', 'Bhubaneswar',
            'Vijayawada Junction', 'Coimbatore Junction', 'Kochi Central',
            'Thiruvananthapuram Central', 'Mysore Junction', 'Mangalore Central',
            'Hubli Junction', 'Belgaum', 'Kolhapur', 'Sangli', 'Satara', 'Pune',
            'Nashik Road', 'Bhusawal', 'Jalgaon', 'Dhule', 'Manmad', 'Aurangabad'
        ]
        
        for i, name in enumerate(station_names):
            stations.append({
                'station_id': f'ST{i+1:03d}',
                'station_name': name,
                'station_type': random.choice(['Major', 'Minor', 'Junction', 'Terminal']),
                'platform_count': random.randint(2, 12),
                'station_congestion': random.uniform(0.1, 0.9),
                'latitude': random.uniform(8.0, 37.0),
                'longitude': random.uniform(68.0, 97.0)
            })
        return stations
    
    def _generate_routes(self):
        """Generate realistic route data"""
        routes = []
        for i in range(50):
            start_station = random.choice(self.stations)
            end_station = random.choice([s for s in self.stations if s != start_station])
            
            # Calculate distance based on coordinates (simplified)
            distance = np.sqrt(
                (start_station['latitude'] - end_station['latitude'])**2 +
                (start_station['longitude'] - end_station['longitude'])**2
            ) * 100  # Rough conversion to km
            
            routes.append({
                'route_id': f'RT{i+1:03d}',
                'start_station': start_station['station_id'],
                'end_station': end_station['station_id'],
                'route_length': distance,
                'track_condition': random.choice(['Excellent', 'Good', 'Fair', 'Poor']),
                'gradient': random.uniform(-2.0, 2.0),  # Percentage gradient
                'signal_density': random.uniform(0.5, 3.0),  # Signals per km
                'junction_count': random.randint(0, 10)
            })
        return routes
    
    def _generate_weather_data(self, timestamp):
        """Generate realistic weather data"""
        month = timestamp.month
        
        # Seasonal temperature variation
        base_temp = 25 + 10 * np.sin(2 * np.pi * month / 12)
        temperature = base_temp + random.uniform(-5, 5)
        
        # Humidity inversely related to temperature
        humidity = max(20, min(95, 80 - (temperature - 20) * 2))
        
        # Precipitation probability (higher in monsoon months)
        precip_prob = 0.1 if month not in [6, 7, 8, 9] else 0.4
        precipitation = random.uniform(0, 20) if random.random() < precip_prob else 0
        
        return {
            'temperature': temperature,
            'humidity': humidity,
            'precipitation': precipitation,
            'wind_speed': random.uniform(0, 30),
            'visibility': random.uniform(0.5, 10.0)
        }
    
    def _calculate_base_travel_time(self, route, train_type):
        """Calculate base travel time based on route and train type"""
        base_speed = {
            'Express': 80,
            'Mail': 70,
            'Passenger': 50,
            'Freight': 40,
            'Superfast': 100,
            'Rajdhani': 120,
            'Shatabdi': 110
        }
        
        speed = base_speed.get(train_type, 60)
        
        # Adjust for track condition
        condition_multiplier = {
            'Excellent': 1.0,
            'Good': 1.1,
            'Fair': 1.2,
            'Poor': 1.4
        }
        
        speed *= condition_multiplier.get(route['track_condition'], 1.0)
        
        # Adjust for gradient
        gradient_factor = 1 + abs(route['gradient']) * 0.1
        speed /= gradient_factor
        
        # Calculate base time
        base_time = route['route_length'] / speed * 60  # Convert to minutes
        
        return base_time
    
    def _calculate_station_stop_time(self, station, train_type):
        """Calculate station stop time based on station and train type"""
        base_stop_time = {
            'Express': 2,
            'Mail': 5,
            'Passenger': 10,
            'Freight': 15,
            'Superfast': 1,
            'Rajdhani': 3,
            'Shatabdi': 2
        }
        
        stop_time = base_stop_time.get(train_type, 5)
        
        # Adjust for station type
        station_multiplier = {
            'Major': 1.5,
            'Minor': 0.8,
            'Junction': 2.0,
            'Terminal': 3.0
        }
        
        stop_time *= station_multiplier.get(station['station_type'], 1.0)
        
        # Add congestion factor
        stop_time *= (1 + station['station_congestion'])
        
        return stop_time
    
    def generate_dataset(self):
        """Generate the complete dataset"""
        data = []
        
        for i in range(self.num_records):
            # Random timestamp within last 2 years
            start_time = datetime.now() - timedelta(days=random.randint(1, 730))
            start_time = start_time.replace(
                hour=random.randint(0, 23),
                minute=random.randint(0, 59),
                second=0,
                microsecond=0
            )
            
            # Select random route and train
            route = random.choice(self.routes)
            train_type = random.choice(self.train_types)
            
            # Get station details
            start_station = next(s for s in self.stations if s['station_id'] == route['start_station'])
            end_station = next(s for s in self.stations if s['station_id'] == route['end_station'])
            
            # Generate weather data
            weather = self._generate_weather_data(start_time)
            
            # Calculate base travel time
            base_travel_time = self._calculate_base_travel_time(route, train_type)
            
            # Add random delays
            delay_factors = []
            
            # Weather delay
            if weather['precipitation'] > 10:
                delay_factors.append(random.uniform(1.2, 1.8))
            elif weather['visibility'] < 2:
                delay_factors.append(random.uniform(1.1, 1.4))
            
            # Track condition delay
            if route['track_condition'] == 'Poor':
                delay_factors.append(random.uniform(1.1, 1.3))
            
            # Congestion delay
            if start_station['station_congestion'] > 0.7:
                delay_factors.append(random.uniform(1.1, 1.5))
            
            # Calculate final travel time
            delay_multiplier = np.prod(delay_factors) if delay_factors else 1.0
            actual_travel_time = base_travel_time * delay_multiplier
            
            # Calculate station stop time
            station_stop_time = self._calculate_station_stop_time(end_station, train_type)
            
            # Add random variation to stop time
            stop_time_variation = random.uniform(0.8, 1.2)
            actual_stop_time = station_stop_time * stop_time_variation
            
            # Calculate arrival time
            arrival_time = start_time + timedelta(minutes=actual_travel_time)
            
            # Create record
            record = {
                'record_id': f'R{i+1:06d}',
                'timestamp': start_time,
                'train_id': f'T{random.randint(1000, 9999)}',
                'train_type': train_type,
                'priority': random.randint(1, 5),
                'start_station_id': route['start_station'],
                'start_station_name': start_station['station_name'],
                'end_station_id': route['end_station'],
                'end_station_name': end_station['station_name'],
                'route_id': route['route_id'],
                'route_length': route['route_length'],
                'track_condition': route['track_condition'],
                'gradient': route['gradient'],
                'signal_density': route['signal_density'],
                'junction_count': route['junction_count'],
                'start_station_type': start_station['station_type'],
                'start_platform_count': start_station['platform_count'],
                'start_station_congestion': start_station['station_congestion'],
                'end_station_type': end_station['station_type'],
                'end_platform_count': end_station['platform_count'],
                'end_station_congestion': end_station['station_congestion'],
                'temperature': weather['temperature'],
                'humidity': weather['humidity'],
                'precipitation': weather['precipitation'],
                'wind_speed': weather['wind_speed'],
                'visibility': weather['visibility'],
                'hour_of_day': start_time.hour,
                'day_of_week': start_time.weekday(),
                'month': start_time.month,
                'is_weekend': start_time.weekday() >= 5,
                'is_holiday': random.random() < 0.05,  # 5% chance of holiday
                'current_speed': random.uniform(30, 120),
                'previous_delay': random.uniform(0, 30),
                'base_travel_time': base_travel_time,
                'actual_travel_time': actual_travel_time,
                'base_stop_time': station_stop_time,
                'actual_stop_time': actual_stop_time,
                'arrival_time': arrival_time,
                'delay': actual_travel_time - base_travel_time
            }
            
            data.append(record)
        
        return pd.DataFrame(data)
    
    def save_dataset(self, df, filename):
        """Save dataset to CSV file"""
        filepath = DATA_CONFIG['dataset_path'] + filename
        df.to_csv(filepath, index=False)
        print(f"Dataset saved to {filepath}")
        return filepath

def main():
    """Generate and save the dataset"""
    print("Generating synthetic railway dataset...")
    
    generator = RailwayDataGenerator(num_records=10000)
    df = generator.generate_dataset()
    
    print(f"Generated {len(df)} records")
    print(f"Columns: {list(df.columns)}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # Save dataset
    generator.save_dataset(df, 'train_data.csv')
    
    # Display sample data
    print("\nSample data:")
    print(df.head())
    
    # Display basic statistics
    print("\nBasic statistics:")
    print(df[['actual_travel_time', 'actual_stop_time', 'delay']].describe())

if __name__ == "__main__":
    main()
