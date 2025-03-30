# Heart Disease Prediction Project

A comprehensive end-to-end machine learning project that predicts heart disease risk using the UCI Heart Disease dataset, featuring a modular pipeline architecture and web-based deployment.

## Table of Contents
- [Project Overview](#project-overview)
- [Installation](#installation)
- [Dataset Description](#dataset-description)
- [Pipeline Architecture](#pipeline-architecture)
- [Web Application](#web-application)
- [API Reference](#api-reference)
- [Model Performance](#model-performance)
- [Project Structure](#project-structure)
- [Advanced Usage](#advanced-usage)
- [Troubleshooting](#troubleshooting)
- [Future Improvements](#future-improvements)

## Project Overview

This project implements a complete machine learning solution for predicting heart disease risk using medical data. It follows a modular pipeline approach with separate stages for data ingestion, validation, transformation, model training, and deployment.

Key features:
- End-to-end ML pipeline with modular components
- Automated data quality validation
- Advanced model training with hyperparameter optimization
- Model comparison and ensemble techniques
- Interactive web interface for predictions
- RESTful API for integration with other systems
- Comprehensive documentation and logging

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)
- Git (optional, for cloning the repository)

### Setup

1. Clone the repository (or download the zip file):
```bash
git clone https://github.com/DRuanli/Heart-Disease.git
cd Heart-Disease
```

2. Create and activate a virtual environment:
```bash
# On Windows
python -m venv .venv
.venv\Scripts\activate

# On macOS/Linux
python -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset Description

This project uses the UCI Heart Disease dataset, which contains medical data from several hospitals across different countries.

### Data Sources
- Cleveland Clinic Foundation
- Hungarian Institute of Cardiology
- V.A. Medical Center, Long Beach
- University Hospital, Switzerland

### Features
The dataset includes 13 features:
- **age**: Age in years
- **sex**: Gender (1 = male, 0 = female)
- **cp**: Chest pain type (0-3)
  - 0: Typical angina
  - 1: Atypical angina
  - 2: Non-anginal pain
  - 3: Asymptomatic
- **trestbps**: Resting blood pressure in mm Hg
- **chol**: Serum cholesterol in mg/dl
- **fbs**: Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)
- **restecg**: Resting electrocardiographic results (0-2)
  - 0: Normal
  - 1: ST-T wave abnormality
  - 2: Left ventricular hypertrophy
- **thalach**: Maximum heart rate achieved
- **exang**: Exercise induced angina (1 = yes, 0 = no)
- **oldpeak**: ST depression induced by exercise relative to rest
- **slope**: Slope of the peak exercise ST segment (0-2)
  - 0: Upsloping
  - 1: Flat
  - 2: Downsloping
- **ca**: Number of major vessels colored by fluoroscopy (0-3)
- **thal**: Thalassemia (0-3)
  - 0: Normal
  - 1: Fixed defect
  - 2: Reversible defect
  - 3: Unknown

### Target Variable
- **target**: Presence of heart disease (0 = no disease, 1-4 = disease with increasing severity)

## Pipeline Architecture

The project is organized into five stages, each implemented as a separate pipeline component:

### Stage 1: Data Ingestion
- Downloads the dataset from the source URL
- Extracts and prepares data files for processing
- Stores the raw data in a consistent format

### Stage 2: Data Validation
- Validates the structure and presence of required files
- Checks data quality and completeness
- Reports validation issues without halting the pipeline
- Creates a validation report for reference

### Stage 3: Data Transformation
- Handles missing values using appropriate imputation strategies
- Applies feature scaling (StandardScaler) for numerical features
- Performs one-hot encoding for categorical features
- Creates a preprocessing pipeline for reproducible transformations
- Splits data into training and test sets

### Stage 4: Model Training
- Implements multiple model types (Random Forest, Gradient Boosting, etc.)
- Performs hyperparameter optimization using Optuna
- Handles class imbalance with SMOTE
- Creates ensemble models for improved performance
- Evaluates models using cross-validation
- Selects the best performing model based on F1 score
- Analyzes feature importance

### Stage 5: Model Deployment
- Creates a prediction pipeline that encapsulates the model and preprocessing
- Provides a web interface for interactive predictions
- Exposes RESTful API endpoints for programmatic access
- Includes detailed feature descriptions for interpretability

## Web Application

The project includes a Flask web application for making predictions through a user-friendly interface.

### Running the Web App

After completing the pipeline stages:

```bash
python app.py
```

By default, the application runs on http://localhost:5002

### Web Interface
The web interface provides:
- A form to input patient data
- Descriptions for each input field
- Prediction results with confidence scores
- Option to make multiple predictions

## API Reference

### Prediction API

#### Single Prediction
- **Endpoint**: `/api/predict`
- **Method**: POST
- **Input**: JSON object with feature values
- **Output**: Prediction with probability

Example request:
```json
{
  "age": 63,
  "sex": 1,
  "cp": 3,
  "trestbps": 145,
  "chol": 233,
  "fbs": 1,
  "restecg": 0,
  "thalach": 150,
  "exang": 0,
  "oldpeak": 2.3,
  "slope": 0,
  "ca": 0,
  "thal": 1
}
```

Example response:
```json
{
  "prediction": 1,
  "prediction_label": "Heart Disease Stage 1",
  "probability": 0.85
}
```

#### Batch Prediction
- **Endpoint**: `/api/batch_predict`
- **Method**: POST
- **Input**: JSON array of objects with feature values
- **Output**: Array of predictions with probabilities

## Model Performance

The project implements multiple models and selects the best performing one. Typical performance metrics include:

- **Accuracy**: Overall correctness of predictions
- **Precision**: Proportion of positive identifications that were actually correct
- **Recall**: Proportion of actual positives that were identified correctly
- **F1 Score**: Harmonic mean of precision and recall
- **ROC AUC**: Area under the Receiver Operating Characteristic curve

Model performances are logged and saved in `artifacts/model_trainer/metrics.json`.

### Feature Importance

The system analyzes which features have the strongest impact on predictions. This information is valuable for clinical interpretability and understanding the model's decision-making process.

## Project Structure

```
Heart-Disease/
├── .venv/                      # Virtual environment (created during setup)
├── artifacts/                  # Generated outputs and models (created during execution)
│   ├── data_ingestion/         # Downloaded and extracted data
│   ├── data_validation/        # Validation reports
│   ├── data_transformation/    # Transformed data and preprocessor
│   ├── model_trainer/          # Trained models and metrics
│   └── model_deployment/       # Deployment artifacts
├── config/                     # Configuration files
│   ├── config.yaml             # Pipeline configuration
│   └── schema.yaml             # Data schema
├── research/                   # Exploratory notebooks
├── src/                        # Source code
│   └── HeartDisease/           # Main package
│       ├── components/         # Pipeline components
│       ├── config/             # Configuration utilities
│       ├── constants/          # Constants and paths
│       ├── entity/             # Data classes and entities
│       ├── pipeline/           # Pipeline orchestration
│       └── utils/              # Utility functions
├── templates/                  # Web application templates
├── .gitignore                  # Git ignore file
├── README.md                   # Project documentation
├── app.py                      # Web application
├── main.py                     # Main execution script
├── params.yaml                 # Model parameters
├── requirements.txt            # Dependencies
└── setup.py                    # Package setup
```

## Advanced Usage

### Running Individual Pipeline Stages

Each stage can be run independently:

```bash
# Run data ingestion
python -m src.HeartDisease.pipeline.stage_01_data_ingestion

# Run data validation
python -m src.HeartDisease.pipeline.stage_02_data_validation

# Run data transformation
python -m src.HeartDisease.pipeline.stage_03_data_transformation

# Run model training
python -m src.HeartDisease.pipeline.stage_04_advanced_model_trainer

# Run model deployment
python -m src.HeartDisease.pipeline.stage_05_model_deployment
```

### Customizing Model Parameters

Model parameters can be customized in `params.yaml`. For example:

```yaml
# Parameters for Random Forest
models:
  RandomForest:
    n_estimators: 200
    max_depth: 15
    min_samples_split: 5
    min_samples_leaf: 2
```

### Adding New Models

To add a new model:

1. Update the `params.yaml` with parameters for the new model
2. Modify `src/HeartDisease/components/advanced_model_trainer.py` to include the new model type
3. Update the model comparison logic to include the new model

## Troubleshooting

### Common Issues

1. **Missing templates error**: 
   - Check that the templates directory exists in the project root
   - Verify that all template files (index.html, result.html, error.html) are present

2. **Model loading errors**:
   - Ensure all pipeline stages have been run successfully
   - Check that the model files exist in the expected locations

3. **Missing dependencies**:
   - Run `pip install -r requirements.txt` to ensure all dependencies are installed
   - Some systems may require additional system libraries for scikit-learn dependencies

### Logging

The project implements comprehensive logging. Check the console output for detailed information about each step of the pipeline.

## Future Improvements

Potential enhancements for the project:

1. **Additional Models**: Implement deep learning models like neural networks
2. **Expanded Dataset**: Incorporate additional heart disease datasets for improved generalization
3. **Feature Engineering**: Create more sophisticated features from the raw data
4. **Explainability**: Add SHAP or LIME for detailed model explanations
5. **Monitoring**: Add model monitoring to track performance over time
6. **Database Integration**: Store predictions and results in a database
7. **User Management**: Add user accounts for tracking patient histories
8. **Containerization**: Package the application using Docker for easier deployment
9. **Cloud Deployment**: Instructions for deploying to cloud platforms

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- UCI Machine Learning Repository for providing the heart disease dataset
- The scikit-learn team for their excellent machine learning library
- Optuna developers for the hyperparameter optimization framework