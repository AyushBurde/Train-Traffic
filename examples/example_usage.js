/**
 * Example usage of Train Arrival Time Prediction System - JavaScript
 */

// Single prediction example
async function singlePrediction() {
    console.log("Single Train Prediction Example");
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
            
            console.log(`Route: ${trainData.start_station_name} -> ${trainData.end_station_name}`);
            console.log(`Train Type: ${trainData.train_type}`);
            console.log(`Travel Time: ${prediction.travel_time_minutes.toFixed(1)} minutes`);
            console.log(`Stop Duration: ${prediction.stop_duration_minutes.toFixed(1)} minutes`);
            console.log(`Arrival Time: ${prediction.arrival_time}`);
            console.log(`Departure Time: ${prediction.departure_time}`);
        } else {
            console.log(`Error: ${response.status} - ${await response.text()}`);
        }
    } catch (error) {
        console.log(`Error: ${error.message}`);
        console.log("Make sure the API server is running: python prediction_api.py");
    }
}

// Batch prediction example
async function batchPrediction() {
    console.log("\nBatch Prediction Example");
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
            
            console.log(`Batch prediction completed!`);
            console.log(`Successful predictions: ${result.successful_predictions}/${result.total_trains}`);
            console.log();
            
            result.results.forEach((trainResult, index) => {
                if (trainResult.status === 'success') {
                    const pred = trainResult.prediction;
                    console.log(`Train ${index + 1}: ${pred.travel_time_minutes.toFixed(1)}min travel + ${pred.stop_duration_minutes.toFixed(1)}min stop`);
                } else {
                    console.log(`Train ${index + 1}: Error - ${trainResult.error}`);
                }
            });
        } else {
            console.log(`Error: ${response.status} - ${await response.text()}`);
        }
    } catch (error) {
        console.log(`Error: ${error.message}`);
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
