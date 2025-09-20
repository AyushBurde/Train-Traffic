# API Reference

## Base URL
```
http://localhost:5000
```

## Authentication
Currently, no authentication is required. All endpoints are publicly accessible.

## Endpoints

### Health Check

#### GET /health
Check if the API is running and models are loaded.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "models_loaded": true
}
```

### Model Information

#### GET /model/info
Get information about loaded models.

**Response:**
```json
{
  "travel_time_model": {
    "type": "XGBRegressor",
    "loaded": true
  },
  "stop_duration_model": {
    "type": "LGBMRegressor", 
    "loaded": true
  },
  "preprocessor_loaded": true,
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### Predictions

#### POST /predict/travel_time
Predict travel time between two stations.

**Request Body:**
```json
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
}
```

**Response:**
```json
{
  "travel_time_minutes": 847.2,
  "travel_time_hours": 14.12,
  "timestamp": "2024-01-15T10:30:00Z"
}
```

#### POST /predict/stop_duration
Predict stop duration at destination station.

**Request Body:**
```json
{
  "end_station_name": "Delhi Junction",
  "train_type": "Rajdhani",
  "priority": 5,
  "temperature": 25.0,
  "humidity": 60.0,
  "precipitation": 0.0,
  "visibility": 10.0,
  "track_condition": "Excellent"
}
```

**Response:**
```json
{
  "stop_duration_minutes": 8.7,
  "timestamp": "2024-01-15T10:30:00Z"
}
```

#### POST /predict/complete
Predict both travel time and stop duration.

**Request Body:**
```json
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
}
```

**Response:**
```json
{
  "prediction": {
    "travel_time_minutes": 847.2,
    "stop_duration_minutes": 8.7,
    "arrival_time": "2024-01-15T18:30:00Z",
    "departure_time": "2024-01-15T18:38:42Z",
    "total_journey_time_minutes": 855.9
  },
  "timestamp": "2024-01-15T10:30:00Z"
}
```

#### POST /predict/batch
Predict for multiple trains in a single request.

**Request Body:**
```json
{
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
```

**Response:**
```json
{
  "results": [
    {
      "train_index": 0,
      "prediction": {
        "travel_time_minutes": 847.2,
        "stop_duration_minutes": 8.7,
        "arrival_time": "2024-01-15T18:30:00Z",
        "departure_time": "2024-01-15T18:38:42Z",
        "total_journey_time_minutes": 855.9
      },
      "status": "success"
    },
    {
      "train_index": 1,
      "prediction": {
        "travel_time_minutes": 245.6,
        "stop_duration_minutes": 12.3,
        "arrival_time": "2024-01-15T14:15:36Z",
        "departure_time": "2024-01-15T14:27:54Z",
        "total_journey_time_minutes": 257.9
      },
      "status": "success"
    }
  ],
  "total_trains": 2,
  "successful_predictions": 2,
  "timestamp": "2024-01-15T10:30:00Z"
}
```

## Request Parameters

### Required Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `start_station_name` | string | Name of the starting station |
| `end_station_name` | string | Name of the destination station |
| `route_length` | number | Distance between stations in kilometers |
| `train_type` | string | Type of train (Express, Rajdhani, Shatabdi, etc.) |

### Optional Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `priority` | integer | 3 | Train priority (1-5) |
| `current_speed` | number | 80.0 | Current speed in km/h |
| `temperature` | number | 25.0 | Temperature in Celsius |
| `humidity` | number | 60.0 | Humidity percentage |
| `precipitation` | number | 0.0 | Precipitation in mm |
| `visibility` | number | 10.0 | Visibility in km |
| `track_condition` | string | "Good" | Track condition (Excellent, Good, Fair, Poor) |
| `gradient` | number | 0.0 | Track gradient percentage |
| `timestamp` | string | current time | ISO timestamp |

## Response Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad Request - Missing or invalid parameters |
| 500 | Internal Server Error - Model or processing error |

## Error Responses

### 400 Bad Request
```json
{
  "error": "Missing required field: start_station_name"
}
```

### 500 Internal Server Error
```json
{
  "error": "Failed to make prediction"
}
```

## Example Usage

### Python
```python
import requests

# Single prediction
data = {
    "start_station_name": "Mumbai Central",
    "end_station_name": "Delhi Junction",
    "route_length": 1384.0,
    "train_type": "Rajdhani"
}

response = requests.post("http://localhost:5000/predict/complete", json=data)
result = response.json()

print(f"Travel Time: {result['prediction']['travel_time_minutes']:.1f} minutes")
print(f"Arrival Time: {result['prediction']['arrival_time']}")
```

### JavaScript
```javascript
const response = await fetch('http://localhost:5000/predict/complete', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    start_station_name: 'Mumbai Central',
    end_station_name: 'Delhi Junction',
    route_length: 1384.0,
    train_type: 'Rajdhani'
  })
});

const result = await response.json();
console.log(`Travel Time: ${result.prediction.travel_time_minutes} minutes`);
```

### cURL
```bash
curl -X POST http://localhost:5000/predict/complete \
  -H "Content-Type: application/json" \
  -d '{
    "start_station_name": "Mumbai Central",
    "end_station_name": "Delhi Junction",
    "route_length": 1384.0,
    "train_type": "Rajdhani"
  }'
```

## Rate Limiting

Currently, no rate limiting is implemented. For production deployment, consider implementing rate limiting to prevent abuse.

## CORS

CORS is enabled for all origins. For production deployment, restrict CORS to specific domains.

## Model Performance

- **Travel Time Model**: XGBoost with 97.5% accuracy (MAE: 83.81 minutes)
- **Stop Duration Model**: LightGBM with 96.9% accuracy (MAE: 1.63 minutes)
- **Features**: 53 engineered features
- **Training Data**: 10,000 synthetic railway records
