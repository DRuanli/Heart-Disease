import os
import sys
import joblib
import json
import pandas as pd
import numpy as np
from flask import Flask, request, render_template, jsonify
import logging

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.HeartDisease.pipeline.stage_05_model_deployment import main as deploy_model
from src.HeartDisease.components.model_deployment import PredictionPipeline
from src.HeartDisease.config.configuration import ConfigurationManager

app = Flask(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s: %(levelname)s]: %(message)s"
)


# Load prediction pipeline
def load_pipeline():
    try:
        config = ConfigurationManager()
        model_deployment_config = config.get_model_deployment_config()

        # Check if prediction pipeline exists
        if not os.path.exists(model_deployment_config.prediction_pipeline_path):
            # Create and save prediction pipeline
            pipeline = deploy_model()
        else:
            # Load prediction pipeline
            pipeline = joblib.load(model_deployment_config.prediction_pipeline_path)

        return pipeline
    except Exception as e:
        logging.error(f"Error loading prediction pipeline: {str(e)}")
        return None


# Initialize prediction pipeline
prediction_pipeline = load_pipeline()


@app.route('/')
def home():
    """
    Render home page with prediction form.
    """
    # Get feature descriptions from pipeline
    feature_descriptions = prediction_pipeline.get_feature_descriptions()

    return render_template('index.html', feature_descriptions=feature_descriptions)


@app.route('/predict', methods=['POST'])
def predict():
    """
    Make a prediction based on form input.
    """
    try:
        # Get form data
        features = {}
        for feature in prediction_pipeline.get_feature_names():
            value = request.form.get(feature)
            if value is not None and value.strip() != '':
                # Convert to appropriate type
                if feature in prediction_pipeline.numerical_columns:
                    features[feature] = float(value)
                else:
                    features[feature] = int(value)

        # Make prediction
        result = prediction_pipeline.predict(features)

        return render_template('result.html', result=result, features=features)

    except Exception as e:
        logging.error(f"Error making prediction: {str(e)}")
        return render_template('error.html', error=str(e))


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    API endpoint for prediction.
    """
    try:
        # Get JSON data
        features = request.json

        # Make prediction
        result = prediction_pipeline.predict(features)

        return jsonify(result)

    except Exception as e:
        logging.error(f"Error making prediction: {str(e)}")
        return jsonify({"error": str(e)}), 400


@app.route('/api/batch_predict', methods=['POST'])
def api_batch_predict():
    """
    API endpoint for batch prediction.
    """
    try:
        # Get JSON data
        data = request.json

        # Convert to DataFrame
        df = pd.DataFrame(data)

        # Make batch prediction
        result_df = prediction_pipeline.batch_predict(df)

        # Convert to dictionary for JSON response
        results = result_df.to_dict(orient='records')

        return jsonify(results)

    except Exception as e:
        logging.error(f"Error making batch prediction: {str(e)}")
        return jsonify({"error": str(e)}), 400


if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)

    # Create index.html if it doesn't exist
    if not os.path.exists('templates/index.html'):
        with open('templates/index.html', 'w') as f:
            f.write("""<!DOCTYPE html>
<html>
<head>
    <title>Heart Disease Prediction</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 5px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }
        h1 {
            color: #333;
            text-align: center;
        }
        .form-group {
            margin-bottom: 15px;
        }
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
        }
        input[type="number"] {
            width: 100%;
            padding: 8px;
            border: 1px solid #ddd;
            border-radius: 4px;
            box-sizing: border-box;
        }
        .description {
            font-size: 0.8em;
            color: #666;
            margin-top: 2px;
        }
        button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 15px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
        }
        button:hover {
            background-color: #45a049;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Heart Disease Prediction</h1>
        <form action="/predict" method="post">
            {% for feature, description in feature_descriptions.items() %}
            <div class="form-group">
                <label for="{{ feature }}">{{ feature }}</label>
                <input type="number" id="{{ feature }}" name="{{ feature }}" step="any" required>
                <div class="description">{{ description }}</div>
            </div>
            {% endfor %}
            <button type="submit">Predict</button>
        </form>
    </div>
</body>
</html>""")

    # Create result.html if it doesn't exist
    if not os.path.exists('templates/result.html'):
        with open('templates/result.html', 'w') as f:
            f.write("""<!DOCTYPE html>
<html>
<head>
    <title>Prediction Result</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 5px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }
        h1 {
            color: #333;
            text-align: center;
        }
        .result {
            margin: 20px 0;
            padding: 15px;
            border-radius: 4px;
            background-color: #e8f5e9;
            border-left: 5px solid #4CAF50;
        }
        .result.negative {
            background-color: #fff3e0;
            border-left: 5px solid #ff9800;
        }
        p {
            margin: 10px 0;
        }
        .probability {
            font-weight: bold;
        }
        .features {
            margin-top: 20px;
            border-top: 1px solid #ddd;
            padding-top: 20px;
        }
        .feature {
            display: flex;
            justify-content: space-between;
            margin-bottom: 5px;
        }
        .back-button {
            display: block;
            margin: 20px auto;
            padding: 10px 15px;
            background-color: #2196F3;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            text-decoration: none;
            text-align: center;
            width: 150px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Prediction Result</h1>
        <div class="result {% if result.prediction == 0 %}negative{% endif %}">
            <h2>{{ result.prediction_label }}</h2>
            <p class="probability">Confidence: {{ "%.2f"|format(result.probability * 100) }}%</p>
        </div>

        <div class="features">
            <h3>Input Features:</h3>
            {% for feature, value in features.items() %}
            <div class="feature">
                <span>{{ feature }}:</span>
                <span>{{ value }}</span>
            </div>
            {% endfor %}
        </div>

        <a href="/" class="back-button">Make Another Prediction</a>
    </div>
</body>
</html>""")

    # Create error.html if it doesn't exist
    if not os.path.exists('templates/error.html'):
        with open('templates/error.html', 'w') as f:
            f.write("""<!DOCTYPE html>
<html>
<head>
    <title>Error</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 5px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }
        h1 {
            color: #f44336;
            text-align: center;
        }
        .error-message {
            margin: 20px 0;
            padding: 15px;
            border-radius: 4px;
            background-color: #ffebee;
            border-left: 5px solid #f44336;
        }
        .back-button {
            display: block;
            margin: 20px auto;
            padding: 10px 15px;
            background-color: #2196F3;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            text-decoration: none;
            text-align: center;
            width: 150px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Error</h1>
        <div class="error-message">
            <p>{{ error }}</p>
        </div>

        <a href="/" class="back-button">Go Back</a>
    </div>
</body>
</html>""")

    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5002)