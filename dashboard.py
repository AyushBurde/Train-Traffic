"""
Interactive Dashboard for Train Arrival Time Predictions
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json
from datetime import datetime, timedelta
import time

# Page configuration
st.set_page_config(
    page_title="Train Arrival Time Predictor",
    page_icon="🚂",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .prediction-card {
        background-color: #e8f4fd;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 2px solid #1f77b4;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class TrainPredictionDashboard:
    def __init__(self):
        self.api_base_url = "http://localhost:5000"
        self.stations = [
            "Mumbai Central", "Delhi Junction", "Chennai Central", "Kolkata Howrah",
            "Bangalore City", "Hyderabad Deccan", "Pune Junction", "Ahmedabad",
            "Jaipur Junction", "Lucknow Charbagh", "Patna Junction", "Bhubaneswar",
            "Vijayawada Junction", "Coimbatore Junction", "Kochi Central",
            "Thiruvananthapuram Central", "Mysore Junction", "Mangalore Central"
        ]
        self.train_types = ["Express", "Mail", "Passenger", "Freight", "Superfast", "Rajdhani", "Shatabdi"]
    
    def check_api_health(self):
        """Check if API is running"""
        try:
            response = requests.get(f"{self.api_base_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def make_prediction(self, data):
        """Make prediction using API"""
        try:
            response = requests.post(f"{self.api_base_url}/predict/complete", json=data, timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"API Error: {response.status_code}"}
        except Exception as e:
            return {"error": f"Connection Error: {str(e)}"}
    
    def get_model_info(self):
        """Get model information"""
        try:
            response = requests.get(f"{self.api_base_url}/model/info", timeout=5)
            if response.status_code == 200:
                return response.json()
            else:
                return None
        except:
            return None

def main():
    dashboard = TrainPredictionDashboard()
    
    # Header
    st.markdown('<h1 class="main-header">🚂 Train Arrival Time Predictor</h1>', unsafe_allow_html=True)
    
    # Check API status
    api_status = dashboard.check_api_health()
    
    if not api_status:
        st.error("⚠️ API is not running. Please start the API server first by running: python prediction_api.py")
        st.stop()
    
    # Sidebar
    st.sidebar.title("🔧 Configuration")
    
    # Model info
    model_info = dashboard.get_model_info()
    if model_info:
        st.sidebar.success("✅ Models loaded successfully")
        st.sidebar.info(f"Travel Time Model: {model_info['travel_time_model']['type']}")
        st.sidebar.info(f"Stop Duration Model: {model_info['stop_duration_model']['type']}")
    else:
        st.sidebar.error("❌ Failed to load model information")
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Single Prediction", "📊 Batch Prediction", "📈 Analytics", "ℹ️ About"])
    
    with tab1:
        st.header("Single Train Prediction")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Route Information")
            start_station = st.selectbox("Start Station", dashboard.stations, index=0)
            end_station = st.selectbox("End Station", dashboard.stations, index=1)
            route_length = st.number_input("Route Length (km)", min_value=1.0, max_value=2000.0, value=500.0, step=10.0)
            
            st.subheader("Train Information")
            train_type = st.selectbox("Train Type", dashboard.train_types, index=0)
            priority = st.slider("Priority (1-5)", 1, 5, 3)
            current_speed = st.number_input("Current Speed (km/h)", min_value=10.0, max_value=200.0, value=80.0, step=5.0)
        
        with col2:
            st.subheader("Environmental Conditions")
            temperature = st.number_input("Temperature (°C)", min_value=-10.0, max_value=50.0, value=25.0, step=1.0)
            humidity = st.slider("Humidity (%)", 0, 100, 60)
            precipitation = st.number_input("Precipitation (mm)", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
            visibility = st.number_input("Visibility (km)", min_value=0.1, max_value=20.0, value=10.0, step=0.1)
            
            st.subheader("Track Conditions")
            track_condition = st.selectbox("Track Condition", ["Excellent", "Good", "Fair", "Poor"], index=1)
            gradient = st.number_input("Gradient (%)", min_value=-5.0, max_value=5.0, value=0.0, step=0.1)
        
        # Prediction button
        if st.button("🚀 Predict Arrival Time", type="primary"):
            with st.spinner("Making prediction..."):
                prediction_data = {
                    "start_station_name": start_station,
                    "end_station_name": end_station,
                    "route_length": route_length,
                    "train_type": train_type,
                    "priority": priority,
                    "current_speed": current_speed,
                    "temperature": temperature,
                    "humidity": humidity,
                    "precipitation": precipitation,
                    "visibility": visibility,
                    "track_condition": track_condition,
                    "gradient": gradient,
                    "timestamp": datetime.now().isoformat()
                }
                
                result = dashboard.make_prediction(prediction_data)
                
                if "error" in result:
                    st.error(f"❌ Prediction failed: {result['error']}")
                else:
                    prediction = result['prediction']
                    
                    # Display results
                    st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
                    st.success("✅ Prediction completed successfully!")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "Travel Time",
                            f"{prediction['travel_time_minutes']:.1f} min",
                            f"{prediction['travel_time_minutes']/60:.1f} hours"
                        )
                    
                    with col2:
                        st.metric(
                            "Stop Duration",
                            f"{prediction['stop_duration_minutes']:.1f} min"
                        )
                    
                    with col3:
                        st.metric(
                            "Total Journey Time",
                            f"{prediction['total_journey_time_minutes']:.1f} min",
                            f"{prediction['total_journey_time_minutes']/60:.1f} hours"
                        )
                    
                    # Arrival and departure times
                    st.subheader("Schedule")
                    arrival_time = datetime.fromisoformat(prediction['arrival_time'])
                    departure_time = datetime.fromisoformat(prediction['departure_time'])
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.info(f"🕐 Arrival Time: {arrival_time.strftime('%Y-%m-%d %H:%M:%S')}")
                    with col2:
                        st.info(f"🚂 Departure Time: {departure_time.strftime('%Y-%m-%d %H:%M:%S')}")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.header("Batch Prediction")
        
        # Sample data for batch prediction
        st.subheader("Upload Train Data")
        
        # Create sample data
        sample_data = {
            "trains": [
                {
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
                },
                {
                    "start_station_name": "Chennai Central",
                    "end_station_name": "Bangalore City",
                    "route_length": 362.0,
                    "train_type": "Express",
                    "priority": 3,
                    "current_speed": 80.0,
                    "temperature": 28.0,
                    "humidity": 70.0,
                    "precipitation": 5.0,
                    "visibility": 8.0,
                    "track_condition": "Good",
                    "gradient": 1.0
                },
                {
                    "start_station_name": "Kolkata Howrah",
                    "end_station_name": "Patna Junction",
                    "route_length": 536.0,
                    "train_type": "Passenger",
                    "priority": 2,
                    "current_speed": 60.0,
                    "temperature": 30.0,
                    "humidity": 80.0,
                    "precipitation": 10.0,
                    "visibility": 5.0,
                    "track_condition": "Fair",
                    "gradient": -0.5
                }
            ]
        }
        
        if st.button("🚀 Run Batch Prediction", type="primary"):
            with st.spinner("Processing batch prediction..."):
                result = requests.post(f"{dashboard.api_base_url}/predict/batch", json=sample_data, timeout=30)
                
                if result.status_code == 200:
                    data = result.json()
                    
                    st.success(f"✅ Batch prediction completed! {data['successful_predictions']}/{data['total_trains']} predictions successful")
                    
                    # Display results in a table
                    results_data = []
                    for train_result in data['results']:
                        if train_result['status'] == 'success':
                            pred = train_result['prediction']
                            results_data.append({
                                'Train Index': train_result['train_index'],
                                'Travel Time (min)': f"{pred['travel_time_minutes']:.1f}",
                                'Stop Duration (min)': f"{pred['stop_duration_minutes']:.1f}",
                                'Total Time (min)': f"{pred['total_journey_time_minutes']:.1f}",
                                'Arrival Time': pred['arrival_time'][:19],
                                'Departure Time': pred['departure_time'][:19]
                            })
                        else:
                            results_data.append({
                                'Train Index': train_result['train_index'],
                                'Travel Time (min)': 'Error',
                                'Stop Duration (min)': 'Error',
                                'Total Time (min)': 'Error',
                                'Arrival Time': train_result['error'],
                                'Departure Time': 'N/A'
                            })
                    
                    df_results = pd.DataFrame(results_data)
                    st.dataframe(df_results, use_container_width=True)
                    
                    # Visualization
                    if data['successful_predictions'] > 0:
                        st.subheader("Visualization")
                        
                        # Prepare data for visualization
                        successful_results = [r for r in data['results'] if r['status'] == 'success']
                        travel_times = [r['prediction']['travel_time_minutes'] for r in successful_results]
                        stop_durations = [r['prediction']['stop_duration_minutes'] for r in successful_results]
                        train_indices = [r['train_index'] for r in successful_results]
                        
                        # Create subplots
                        fig = make_subplots(
                            rows=1, cols=2,
                            subplot_titles=('Travel Time by Train', 'Stop Duration by Train'),
                            specs=[[{"type": "bar"}, {"type": "bar"}]]
                        )
                        
                        fig.add_trace(
                            go.Bar(x=train_indices, y=travel_times, name='Travel Time', marker_color='lightblue'),
                            row=1, col=1
                        )
                        
                        fig.add_trace(
                            go.Bar(x=train_indices, y=stop_durations, name='Stop Duration', marker_color='lightcoral'),
                            row=1, col=2
                        )
                        
                        fig.update_layout(height=400, showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)
                
                else:
                    st.error(f"❌ Batch prediction failed: {result.status_code}")
    
    with tab3:
        st.header("Analytics Dashboard")
        
        # Load sample data for analytics
        try:
            df = pd.read_csv('data/train_data.csv')
            
            st.subheader("Dataset Overview")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Records", f"{len(df):,}")
            with col2:
                st.metric("Unique Trains", df['train_id'].nunique())
            with col3:
                st.metric("Unique Routes", df['route_id'].nunique())
            with col4:
                st.metric("Date Range", f"{df['timestamp'].min()[:10]} to {df['timestamp'].max()[:10]}")
            
            # Visualizations
            st.subheader("Travel Time Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Travel time by train type
                fig = px.box(df, x='train_type', y='actual_travel_time', 
                           title='Travel Time Distribution by Train Type')
                fig.update_xaxis(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Travel time vs route length
                fig = px.scatter(df.sample(1000), x='route_length', y='actual_travel_time',
                               color='train_type', title='Travel Time vs Route Length')
                st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Stop Duration Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Stop duration by train type
                fig = px.box(df, x='train_type', y='actual_stop_time',
                           title='Stop Duration Distribution by Train Type')
                fig.update_xaxis(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Stop duration vs station congestion
                fig = px.scatter(df.sample(1000), x='end_station_congestion', y='actual_stop_time',
                               color='train_type', title='Stop Duration vs Station Congestion')
                st.plotly_chart(fig, use_container_width=True)
            
            # Weather impact analysis
            st.subheader("Weather Impact Analysis")
            
            # Create weather severity categories
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
            }).reset_index()
            
            weather_impact['weather_category'] = weather_impact['weather_severity'].map({
                0: 'Clear',
                1: 'Mild',
                2: 'Moderate',
                3: 'Severe',
                4: 'Extreme'
            })
            
            fig = px.bar(weather_impact, x='weather_category', y=['actual_travel_time', 'actual_stop_time'],
                        title='Impact of Weather on Travel Time and Stop Duration',
                        barmode='group')
            st.plotly_chart(fig, use_container_width=True)
            
        except FileNotFoundError:
            st.warning("Sample data not found. Please run the data generation script first.")
    
    with tab4:
        st.header("About the System")
        
        st.markdown("""
        ## 🚂 Train Arrival Time Prediction System
        
        This system uses advanced machine learning models to predict:
        - **Travel Time**: How long a train will take to travel from Station A to Station B
        - **Stop Duration**: How long a train will stop at the destination station
        
        ### 🎯 Key Features
        
        - **Real-time Predictions**: Get instant predictions for train arrival times
        - **Multiple ML Models**: Uses XGBoost and LightGBM for high accuracy
        - **Comprehensive Features**: Considers weather, track conditions, train type, and more
        - **Batch Processing**: Predict for multiple trains simultaneously
        - **Interactive Dashboard**: User-friendly interface for predictions and analytics
        
        ### 🔧 Technical Details
        
        - **Travel Time Model**: XGBoost (MAE: 83.81 minutes, R²: 0.975)
        - **Stop Duration Model**: LightGBM (MAE: 1.63 minutes, R²: 0.969)
        - **Features**: 53 engineered features including time, weather, train, station, and route characteristics
        - **API**: RESTful API for integration with other systems
        
        ### 📊 Model Performance
        
        The models achieve high accuracy by considering:
        - Route characteristics (length, gradient, track condition)
        - Train specifications (type, priority, speed)
        - Environmental factors (weather, visibility)
        - Station conditions (congestion, platform count)
        - Temporal patterns (time of day, day of week)
        
        ### 🚀 Usage
        
        1. **Single Prediction**: Use the first tab to predict for one train
        2. **Batch Prediction**: Use the second tab for multiple trains
        3. **Analytics**: View data insights in the third tab
        4. **API Integration**: Use the REST API for system integration
        
        ### 📈 Future Enhancements
        
        - Real-time data integration
        - Advanced weather forecasting
        - Dynamic re-routing suggestions
        - Performance monitoring and alerts
        - Mobile application
        """)

if __name__ == "__main__":
    main()
