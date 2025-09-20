# Deployment Guide

This guide covers different deployment options for the Train Arrival Time Prediction System.

## 🚀 Quick Start (Local Development)

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git

### Steps
```bash
# 1. Clone repository
git clone https://github.com/your-username/train-predictor.git
cd train-predictor

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Generate data and train models
python data_generator.py
python train_models.py

# 5. Start API server
python prediction_api.py

# 6. Launch dashboard (in another terminal)
streamlit run dashboard.py
```

## 🐳 Docker Deployment

### Create Dockerfile
```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data models results

# Generate data and train models
RUN python data_generator.py && python train_models.py

# Expose ports
EXPOSE 5000 8501

# Create startup script
RUN echo '#!/bin/bash\n\
python prediction_api.py &\n\
streamlit run dashboard.py --server.port=8501 --server.address=0.0.0.0\n\
' > start.sh && chmod +x start.sh

CMD ["./start.sh"]
```

### Build and Run
```bash
# Build image
docker build -t train-predictor .

# Run container
docker run -p 5000:5000 -p 8501:8501 train-predictor
```

### Docker Compose
```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "5000:5000"
    environment:
      - FLASK_ENV=production
    volumes:
      - ./models:/app/models
      - ./data:/app/data
    command: python prediction_api.py

  dashboard:
    build: .
    ports:
      - "8501:8501"
    environment:
      - API_URL=http://api:5000
    depends_on:
      - api
    command: streamlit run dashboard.py --server.port=8501 --server.address=0.0.0.0
```

## ☁️ Cloud Deployment

### AWS Deployment

#### Using EC2
1. **Launch EC2 Instance**
   - Choose Ubuntu 20.04 LTS
   - Instance type: t3.medium or larger
   - Security groups: Allow ports 22, 5000, 8501

2. **Install Dependencies**
   ```bash
   sudo apt update
   sudo apt install python3-pip python3-venv git
   ```

3. **Deploy Application**
   ```bash
   git clone https://github.com/your-username/train-predictor.git
   cd train-predictor
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   python data_generator.py
   python train_models.py
   ```

4. **Configure Services**
   ```bash
   # Create systemd service for API
   sudo nano /etc/systemd/system/train-predictor-api.service
   ```
   ```ini
   [Unit]
   Description=Train Predictor API
   After=network.target

   [Service]
   Type=simple
   User=ubuntu
   WorkingDirectory=/home/ubuntu/train-predictor
   Environment=PATH=/home/ubuntu/train-predictor/venv/bin
   ExecStart=/home/ubuntu/train-predictor/venv/bin/python prediction_api.py
   Restart=always

   [Install]
   WantedBy=multi-user.target
   ```

5. **Start Services**
   ```bash
   sudo systemctl enable train-predictor-api
   sudo systemctl start train-predictor-api
   ```

#### Using AWS Lambda + API Gateway
1. **Package for Lambda**
   ```bash
   pip install -r requirements.txt -t lambda-package/
   cp *.py lambda-package/
   cd lambda-package && zip -r ../train-predictor-lambda.zip .
   ```

2. **Create Lambda Function**
   - Runtime: Python 3.9
   - Handler: prediction_api.lambda_handler
   - Upload zip file

3. **Configure API Gateway**
   - Create REST API
   - Configure routes
   - Enable CORS

### Google Cloud Platform

#### Using Cloud Run
1. **Create Dockerfile** (see Docker section)

2. **Deploy to Cloud Run**
   ```bash
   gcloud builds submit --tag gcr.io/PROJECT_ID/train-predictor
   gcloud run deploy --image gcr.io/PROJECT_ID/train-predictor --platform managed
   ```

#### Using Compute Engine
1. **Create VM Instance**
   - Machine type: e2-medium
   - Boot disk: Ubuntu 20.04 LTS

2. **Deploy Application** (similar to AWS EC2)

### Azure Deployment

#### Using Container Instances
1. **Build and Push Image**
   ```bash
   az acr build --registry myregistry --image train-predictor .
   ```

2. **Deploy Container**
   ```bash
   az container create \
     --resource-group myResourceGroup \
     --name train-predictor \
     --image myregistry.azurecr.io/train-predictor \
     --ports 5000 8501
   ```

## 🔧 Production Configuration

### Environment Variables
```bash
# API Configuration
export FLASK_ENV=production
export FLASK_DEBUG=False
export API_HOST=0.0.0.0
export API_PORT=5000

# Model Configuration
export MODEL_PATH=/app/models
export DATA_PATH=/app/data

# Database (if using)
export DATABASE_URL=postgresql://user:pass@localhost/train_predictor
```

### Nginx Configuration
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location /api/ {
        proxy_pass http://localhost:5000/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    location / {
        proxy_pass http://localhost:8501/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### SSL Configuration
```bash
# Install Certbot
sudo apt install certbot python3-certbot-nginx

# Obtain SSL certificate
sudo certbot --nginx -d your-domain.com
```

## 📊 Monitoring and Logging

### Application Logs
```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
```

### Health Checks
```bash
# API health check
curl http://localhost:5000/health

# Model health check
curl http://localhost:5000/model/info
```

### Performance Monitoring
- Use tools like Prometheus + Grafana
- Monitor API response times
- Track model prediction accuracy
- Monitor resource usage

## 🔒 Security Considerations

### API Security
1. **Rate Limiting**
   ```python
   from flask_limiter import Limiter
   
   limiter = Limiter(
       app,
       key_func=lambda: request.remote_addr,
       default_limits=["100 per hour"]
   )
   ```

2. **Input Validation**
   ```python
   from marshmallow import Schema, fields
   
   class PredictionSchema(Schema):
       start_station_name = fields.Str(required=True)
       end_station_name = fields.Str(required=True)
       route_length = fields.Float(required=True, validate=lambda x: x > 0)
   ```

3. **Authentication** (if needed)
   ```python
   from flask_jwt_extended import JWTManager, jwt_required
   
   app.config['JWT_SECRET_KEY'] = 'your-secret-key'
   jwt = JWTManager(app)
   
   @app.route('/predict/complete', methods=['POST'])
   @jwt_required()
   def predict_complete():
       # Your code here
   ```

### Data Security
- Encrypt sensitive data
- Use secure connections (HTTPS)
- Implement proper access controls
- Regular security audits

## 🚀 Scaling Considerations

### Horizontal Scaling
- Use load balancer (nginx, HAProxy)
- Deploy multiple API instances
- Use container orchestration (Kubernetes)

### Vertical Scaling
- Increase server resources
- Optimize model performance
- Use faster hardware (GPU for ML)

### Database Scaling
- Use Redis for caching
- Implement database clustering
- Use read replicas for queries

## 📈 Performance Optimization

### Model Optimization
```python
# Use model quantization
import onnx
from skl2onnx import convert_sklearn

# Convert to ONNX for faster inference
onnx_model = convert_sklearn(travel_model)
```

### Caching
```python
from flask_caching import Cache

cache = Cache(app, config={'CACHE_TYPE': 'simple'})

@cache.memoize(timeout=300)
def predict_travel_time(data):
    # Your prediction code
```

### Async Processing
```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=4)

async def predict_async(data):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, predict_travel_time, data)
```

## 🔄 CI/CD Pipeline

### GitHub Actions
```yaml
name: Deploy to Production

on:
  push:
    branches: [ main ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    
    - name: Deploy to AWS
      run: |
        # Your deployment commands
        aws s3 sync . s3://your-bucket/
        aws lambda update-function-code --function-name train-predictor --s3-bucket your-bucket --s3-key lambda.zip
```

## 📋 Maintenance

### Regular Tasks
- Monitor model performance
- Update dependencies
- Backup models and data
- Review logs for errors
- Update documentation

### Model Retraining
- Schedule regular retraining
- A/B test new models
- Monitor prediction drift
- Update feature engineering

---

For more detailed deployment instructions, refer to the specific cloud provider documentation or contact the development team.
