"""
Script to set up GitHub repository with all documentation and code
"""

import os
import subprocess
import sys
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e.stderr}")
        return None

def create_gitignore():
    """Create .gitignore file"""
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
.hypothesis/
.pytest_cache/

# Jupyter Notebook
.ipynb_checkpoints

# IPython
profile_default/
ipython_config.py

# pyenv
.python-version

# celery beat schedule file
celerybeat-schedule

# SageMath parsed files
*.sage.py

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Spyder project settings
.spyderproject
.spyproject

# mkdocs documentation
/site

# mypy
.mypy_cache/
.dmypy.json
dmypy.json

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Project specific
*.pkl
*.joblib
*.h5
*.hdf5
*.model

# Data files (optional - uncomment if you don't want to include data)
# data/
# *.csv

# Logs
*.log
logs/

# Temporary files
temp/
tmp/
"""
    
    with open('.gitignore', 'w') as f:
        f.write(gitignore_content)
    print("✅ .gitignore created")

def create_repository_structure():
    """Create the complete repository structure"""
    print("🏗️ Creating repository structure...")
    
    # Create directories
    directories = [
        'docs',
        'tests',
        'examples',
        'scripts',
        '.github/workflows',
        'data',
        'models',
        'results'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"📁 Created directory: {directory}")
    
    # Create empty __init__.py files
    init_files = [
        'tests/__init__.py',
        'examples/__init__.py',
        'scripts/__init__.py'
    ]
    
    for init_file in init_files:
        with open(init_file, 'w') as f:
            f.write('')
        print(f"📄 Created: {init_file}")

def create_example_usage():
    """Create example usage files"""
    print("📝 Creating example usage files...")
    
    # Python example
    python_example = '''"""
Example usage of Train Arrival Time Prediction System
"""

import requests
import json
from datetime import datetime

def example_single_prediction():
    """Example of single train prediction"""
    print("🚂 Single Train Prediction Example")
    print("=" * 40)
    
    # API endpoint
    api_url = "http://localhost:5000"
    
    # Sample data
    train_data = {
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
    
    try:
        # Make prediction
        response = requests.post(f"{api_url}/predict/complete", json=train_data)
        
        if response.status_code == 200:
            result = response.json()
            prediction = result['prediction']
            
            print(f"📍 Route: {train_data['start_station_name']} → {train_data['end_station_name']}")
            print(f"🚂 Train Type: {train_data['train_type']}")
            print(f"⏱️ Travel Time: {prediction['travel_time_minutes']:.1f} minutes")
            print(f"🛑 Stop Duration: {prediction['stop_duration_minutes']:.1f} minutes")
            print(f"🕐 Arrival Time: {prediction['arrival_time']}")
            print(f"🚂 Departure Time: {prediction['departure_time']}")
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
    
    except requests.exceptions.ConnectionError:
        print("❌ API server is not running. Please start it with: python prediction_api.py")
    except Exception as e:
        print(f"❌ Error: {str(e)}")

def example_batch_prediction():
    """Example of batch prediction"""
    print("\\n📊 Batch Prediction Example")
    print("=" * 40)
    
    api_url = "http://localhost:5000"
    
    # Multiple trains
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
            },
            {
                "start_station_name": "Kolkata Howrah",
                "end_station_name": "Patna Junction",
                "route_length": 536.0,
                "train_type": "Passenger"
            }
        ]
    }
    
    try:
        response = requests.post(f"{api_url}/predict/batch", json=batch_data)
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"✅ Batch prediction completed!")
            print(f"📈 Successful predictions: {result['successful_predictions']}/{result['total_trains']}")
            print()
            
            for i, train_result in enumerate(result['results']):
                if train_result['status'] == 'success':
                    pred = train_result['prediction']
                    print(f"Train {i+1}: {pred['travel_time_minutes']:.1f}min travel + {pred['stop_duration_minutes']:.1f}min stop")
                else:
                    print(f"Train {i+1}: ❌ {train_result['error']}")
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
    
    except requests.exceptions.ConnectionError:
        print("❌ API server is not running. Please start it with: python prediction_api.py")
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    print("🚂 Train Arrival Time Prediction System - Example Usage")
    print("=" * 60)
    
    # Check if API is running
    try:
        response = requests.get("http://localhost:5000/health", timeout=5)
        if response.status_code == 200:
            print("✅ API is running!")
            example_single_prediction()
            example_batch_prediction()
        else:
            print("❌ API is not responding properly")
    except:
        print("❌ API is not running. Please start it with: python prediction_api.py")
        print("\\nTo run this example:")
        print("1. Start API: python prediction_api.py")
        print("2. Run example: python examples/example_usage.py")
'''
    
    with open('examples/example_usage.py', 'w') as f:
        f.write(python_example)
    
    # JavaScript example
    js_example = '''/**
 * Example usage of Train Arrival Time Prediction System - JavaScript
 */

// Single prediction example
async function singlePrediction() {
    console.log("🚂 Single Train Prediction Example");
    console.log("=" .repeat(40));
    
    const apiUrl = "http://localhost:5000";
    
    const trainData = {
        start_station_name: "Mumbai Central",
        end_station_name: "Delhi Junction",
        route_length: 1384.0,
        train_type: "Rajdhani",
        priority: 5,
        current_speed: 120.0,
        temperature: 25.0,
        humidity: 60.0,
        precipitation: 0.0,
        visibility: 10.0,
        track_condition: "Excellent",
        gradient: 0.0
    };
    
    try {
        const response = await fetch(`${apiUrl}/predict/complete`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(trainData)
        });
        
        if (response.ok) {
            const result = await response.json();
            const prediction = result.prediction;
            
            console.log(`📍 Route: ${trainData.start_station_name} → ${trainData.end_station_name}`);
            console.log(`🚂 Train Type: ${trainData.train_type}`);
            console.log(`⏱️ Travel Time: ${prediction.travel_time_minutes.toFixed(1)} minutes`);
            console.log(`🛑 Stop Duration: ${prediction.stop_duration_minutes.toFixed(1)} minutes`);
            console.log(`🕐 Arrival Time: ${prediction.arrival_time}`);
            console.log(`🚂 Departure Time: ${prediction.departure_time}`);
        } else {
            console.log(`❌ Error: ${response.status} - ${await response.text()}`);
        }
    } catch (error) {
        console.log(`❌ Error: ${error.message}`);
        console.log("Make sure the API server is running: python prediction_api.py");
    }
}

// Batch prediction example
async function batchPrediction() {
    console.log("\\n📊 Batch Prediction Example");
    console.log("=" .repeat(40));
    
    const apiUrl = "http://localhost:5000";
    
    const batchData = {
        trains: [
            {
                start_station_name: "Mumbai Central",
                end_station_name: "Delhi Junction",
                route_length: 1384.0,
                train_type: "Rajdhani"
            },
            {
                start_station_name: "Chennai Central",
                end_station_name: "Bangalore City",
                route_length: 362.0,
                train_type: "Express"
            },
            {
                start_station_name: "Kolkata Howrah",
                end_station_name: "Patna Junction",
                route_length: 536.0,
                train_type: "Passenger"
            }
        ]
    };
    
    try {
        const response = await fetch(`${apiUrl}/predict/batch`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(batchData)
        });
        
        if (response.ok) {
            const result = await response.json();
            
            console.log(`✅ Batch prediction completed!`);
            console.log(`📈 Successful predictions: ${result.successful_predictions}/${result.total_trains}`);
            console.log();
            
            result.results.forEach((trainResult, index) => {
                if (trainResult.status === 'success') {
                    const pred = trainResult.prediction;
                    console.log(`Train ${index + 1}: ${pred.travel_time_minutes.toFixed(1)}min travel + ${pred.stop_duration_minutes.toFixed(1)}min stop`);
                } else {
                    console.log(`Train ${index + 1}: ❌ ${trainResult.error}`);
                }
            });
        } else {
            console.log(`❌ Error: ${response.status} - ${await response.text()}`);
        }
    } catch (error) {
        console.log(`❌ Error: ${error.message}`);
        console.log("Make sure the API server is running: python prediction_api.py");
    }
}

// Run examples
async function runExamples() {
    await singlePrediction();
    await batchPrediction();
}

// Run if this file is executed directly
if (typeof window === 'undefined') {
    runExamples();
}
'''
    
    with open('examples/example_usage.js', 'w') as f:
        f.write(js_example)
    
    print("✅ Example usage files created")

def create_scripts():
    """Create utility scripts"""
    print("🔧 Creating utility scripts...")
    
    # Quick start script
    quick_start = '''#!/bin/bash
# Quick start script for Train Arrival Time Prediction System

echo "🚂 Train Arrival Time Prediction System - Quick Start"
echo "=" .repeat(60)

# Check Python version
python_version=$(python3 --version 2>&1)
echo "🐍 Python version: $python_version"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Generate data if not exists
if [ ! -f "data/train_data.csv" ]; then
    echo "📊 Generating dataset..."
    python data_generator.py
fi

# Train models if not exists
if [ ! -f "models/best_travel_time_model.pkl" ]; then
    echo "🧠 Training ML models..."
    python train_models.py
fi

echo "✅ Setup completed!"
echo ""
echo "🚀 To start the system:"
echo "1. Start API: python prediction_api.py"
echo "2. Launch dashboard: streamlit run dashboard.py"
echo "3. Run demo: python demo.py"
'''
    
    with open('scripts/quick_start.sh', 'w') as f:
        f.write(quick_start)
    
    # Make it executable
    os.chmod('scripts/quick_start.sh', 0o755)
    
    # Windows batch file
    quick_start_bat = '''@echo off
REM Quick start script for Train Arrival Time Prediction System

echo 🚂 Train Arrival Time Prediction System - Quick Start
echo ============================================================

REM Check Python version
python --version
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Check if virtual environment exists
if not exist "venv" (
    echo 📦 Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo 🔄 Activating virtual environment...
call venv\\Scripts\\activate.bat

REM Install dependencies
echo 📥 Installing dependencies...
pip install -r requirements.txt

REM Generate data if not exists
if not exist "data\\train_data.csv" (
    echo 📊 Generating dataset...
    python data_generator.py
)

REM Train models if not exists
if not exist "models\\best_travel_time_model.pkl" (
    echo 🧠 Training ML models...
    python train_models.py
)

echo ✅ Setup completed!
echo.
echo 🚀 To start the system:
echo 1. Start API: python prediction_api.py
echo 2. Launch dashboard: streamlit run dashboard.py
echo 3. Run demo: python demo.py
pause
'''
    
    with open('scripts/quick_start.bat', 'w') as f:
        f.write(quick_start_bat)
    
    print("✅ Utility scripts created")

def main():
    """Main function to set up GitHub repository"""
    print("🚂 Setting up GitHub Repository for Train Arrival Time Prediction System")
    print("=" * 80)
    
    # Create repository structure
    create_repository_structure()
    
    # Create .gitignore
    create_gitignore()
    
    # Create example usage files
    create_example_usage()
    
    # Create utility scripts
    create_scripts()
    
    print("\n✅ Repository setup completed!")
    print("\n📋 Next steps:")
    print("1. Initialize git repository: git init")
    print("2. Add all files: git add .")
    print("3. Create initial commit: git commit -m 'Initial commit: Train Arrival Time Prediction System'")
    print("4. Create GitHub repository on GitHub.com")
    print("5. Add remote origin: git remote add origin https://github.com/your-username/train-predictor.git")
    print("6. Push to GitHub: git push -u origin main")
    print("\n🎉 Your repository is ready for GitHub!")

if __name__ == "__main__":
    main()
