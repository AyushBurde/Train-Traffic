
Frontend UI - https://indian-railways-inte-191c.bolt.host/
# 🚂 Train Arrival Time Prediction System

A comprehensive machine learning system for predicting train arrival times and stop durations using advanced ML algorithms and real-time data processing.

## 🎯 Overview

This system addresses the critical need for precise train traffic control in Indian Railways by providing AI-powered predictions for:
- **Travel Time**: How long a train takes from Station A to Station B
- **Stop Duration**: How long a train stops at the destination station

## ✨ Key Features

- **High-Accuracy ML Models**: XGBoost and LightGBM with 97%+ accuracy
- **Real-time Predictions**: Instant API-based predictions
- **Comprehensive Feature Engineering**: 53+ features including weather, track conditions, and train characteristics
- **Interactive Dashboard**: Streamlit-based web interface
- **Batch Processing**: Predict multiple trains simultaneously
- **RESTful API**: Easy integration with existing systems

## 🏗️ System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Data Source   │───▶│  Preprocessing   │───▶│   ML Models     │
│   (Synthetic)   │    │  & Feature Eng.  │    │  (XGBoost/LGBM) │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Dashboard     │◀───│   Prediction     │◀───│   Model Store   │
│   (Streamlit)   │    │   API (Flask)    │    │   (Pickle)      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 📊 Model Performance

### Travel Time Prediction
- **Best Model**: XGBoost
- **MAE**: 83.81 minutes
- **RMSE**: 157.77 minutes
- **R² Score**: 0.975

### Stop Duration Prediction
- **Best Model**: LightGBM
- **MAE**: 1.63 minutes
- **RMSE**: 2.85 minutes
- **R² Score**: 0.969

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd train-predictor

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Dataset

```bash
python data_generator.py
```

### 3. Train Models

```bash
python train_models.py
```

### 4. Start API Server

```bash
python prediction_api.py
```

### 5. Launch Dashboard

```bash
streamlit run dashboard.py
```

## 📁 Project Structure

```
train-predictor/
├── data/                          # Dataset storage
│   └── train_data.csv            # Generated dataset
├── models/                        # Trained models
│   ├── best_travel_time_model.pkl
│   ├── best_stop_duration_model.pkl
│   └── preprocessor.pkl
├── results/                       # Visualization outputs
├── config.py                     # Configuration settings
├── data_generator.py             # Synthetic data generation
├── data_preprocessing.py         # Feature engineering
├── train_models.py               # Model training
├── prediction_api.py             # REST API server
├── dashboard.py                  # Streamlit dashboard
├── requirements.txt              # Dependencies
└── README.md                     # This file
```

## 🔧 Usage Examples

### API Usage

#### Single Prediction
```python
import requests

# Predict travel time and stop duration
data = {
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

response = requests.post("http://localhost:5000/predict/complete", json=data)
result = response.json()

print(f"Travel Time: {result['prediction']['travel_time_minutes']:.1f} minutes")
print(f"Stop Duration: {result['prediction']['stop_duration_minutes']:.1f} minutes")
print(f"Arrival Time: {result['prediction']['arrival_time']}")
```

#### Batch Prediction
```python
# Predict multiple trains
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

response = requests.post("http://localhost:5000/predict/batch", json=batch_data)
results = response.json()
```

### Direct Model Usage

```python
import joblib
import pandas as pd
from data_preprocessing import DataPreprocessor

# Load models
travel_model = joblib.load('models/best_travel_time_model.pkl')
stop_model = joblib.load('models/best_stop_duration_model.pkl')
preprocessor = DataPreprocessor()
preprocessor.load_preprocessor('models/preprocessor.pkl')

# Prepare data
input_data = {
    "start_station_name": "Mumbai Central",
    "end_station_name": "Delhi Junction",
    "route_length": 1384.0,
    "train_type": "Rajdhani",
    # ... other features
}

# Make prediction
X = preprocessor.prepare_prediction_data(input_data)
travel_time = travel_model.predict(X)[0]
stop_duration = stop_model.predict(X)[0]

print(f"Predicted travel time: {travel_time:.1f} minutes")
print(f"Predicted stop duration: {stop_duration:.1f} minutes")
```

## 📈 Features Used

### Time Features
- Hour of day, day of week, month
- Cyclical encoding (sin/cos)
- Weekend and holiday indicators

### Weather Features
- Temperature, humidity, precipitation
- Wind speed, visibility
- Weather severity indicators

### Train Features
- Train type, priority, capacity
- Current speed, previous delays
- Speed efficiency metrics

### Station Features
- Station type, platform count
- Congestion levels
- Capacity utilization

### Route Features
- Route length, gradient
- Track condition, signal density
- Junction count, complexity

### Interaction Features
- Weather-route interactions
- Train-station priority
- Congestion-delay relationships

## 🔍 Model Details

### XGBoost (Travel Time)
- **Parameters**: n_estimators=50, learning_rate=0.1, max_depth=6
- **Features**: 53 engineered features
- **Performance**: 97.5% accuracy

### LightGBM (Stop Duration)
- **Parameters**: n_estimators=50, learning_rate=0.1, max_depth=6
- **Features**: 53 engineered features
- **Performance**: 96.9% accuracy

## 🌐 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/predict/travel_time` | POST | Predict travel time only |
| `/predict/stop_duration` | POST | Predict stop duration only |
| `/predict/complete` | POST | Predict both travel time and stop duration |
| `/predict/batch` | POST | Batch prediction for multiple trains |
| `/model/info` | GET | Get model information |

## 📊 Dashboard Features

### Single Prediction Tab
- Interactive form for input parameters
- Real-time prediction results
- Visual metrics display

### Batch Prediction Tab
- Sample data for testing
- Results table and visualization
- Performance metrics

### Analytics Tab
- Dataset overview and statistics
- Travel time analysis by train type
- Weather impact visualization
- Stop duration patterns

### About Tab
- System documentation
- Technical specifications
- Usage guidelines

## 🔧 Configuration

Edit `config.py` to customize:
- Dataset parameters
- Model settings
- API configuration
- Feature engineering options

## 📝 Requirements

- Python 3.8+
- pandas, numpy, scikit-learn
- xgboost, lightgbm, catboost
- flask, streamlit
- plotly, matplotlib, seaborn

## 🚀 Deployment

### Local Deployment
1. Install dependencies
2. Generate dataset
3. Train models
4. Start API server
5. Launch dashboard

### Production Deployment
1. Use production WSGI server (gunicorn)
2. Set up reverse proxy (nginx)
3. Configure environment variables
4. Set up monitoring and logging
5. Implement model versioning

## 🔮 Future Enhancements

- **Real-time Data Integration**: Connect to live railway data feeds
- **Advanced Weather Forecasting**: Integrate weather APIs
- **Dynamic Re-routing**: Suggest alternative routes
- **Performance Monitoring**: Real-time model performance tracking
- **Mobile Application**: Native mobile app for controllers
- **Edge Deployment**: Deploy models closer to railway stations

## 📞 Support

For questions or issues:
1. Check the documentation
2. Review the code examples
3. Open an issue in the repository
4. Contact the development team

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Indian Railways for the problem domain
- Open source ML libraries (XGBoost, LightGBM, scikit-learn)
- Streamlit and Flask communities
- Railway operations research community

---

**Built with ❤️ for Indian Railways**

