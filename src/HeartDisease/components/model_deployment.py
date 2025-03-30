import os
import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path
from typing import Dict, List, Union
from src.HeartDisease.entity import ModelDeploymentConfig
from src.HeartDisease.utils import read_yaml


class PredictionPipeline:
    """
    Class to create a prediction pipeline for heart disease prediction.
    """

    def __init__(self, model_path, preprocessor_path, schema_file):
        """
        Initialize the prediction pipeline.

        Args:
            model_path (str): Path to the trained model
            preprocessor_path (str): Path to the preprocessor
            schema_file (str): Path to the schema file
        """
        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(preprocessor_path)
        self.schema = read_yaml(schema_file)
        self.target_column = self.schema.target_column

        # Get column types from schema
        self.numerical_columns = self.schema.numerical_columns
        self.categorical_columns = self.schema.categorical_columns

        # Get target mapping for classification results
        self.target_mapping = {
            0: "No Heart Disease",
            1: "Heart Disease Stage 1",
            2: "Heart Disease Stage 2",
            3: "Heart Disease Stage 3",
            4: "Heart Disease Stage 4"
        }

    def predict(self, features: Dict) -> Dict:
        """
        Make a prediction using the model.

        Args:
            features (Dict): Dictionary of features

        Returns:
            Dict: Prediction result with probability
        """
        try:
            # Convert dictionary to DataFrame
            input_df = pd.DataFrame([features])

            # Apply preprocessing
            input_features = self.preprocessor.transform(input_df)

            # Make prediction
            prediction = self.model.predict(input_features)
            prediction_proba = self.model.predict_proba(input_features)

            # Get the prediction class and probability
            prediction_class = int(prediction[0])
            prediction_probability = float(max(prediction_proba[0]))
            prediction_label = self.target_mapping.get(prediction_class, f"Class {prediction_class}")

            # Return prediction result
            result = {
                "prediction": prediction_class,
                "prediction_label": prediction_label,
                "probability": prediction_probability
            }

            return result

        except Exception as e:
            logging.error(f"Error making prediction: {str(e)}")
            raise e

    def batch_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions for a batch of data.

        Args:
            df (pd.DataFrame): DataFrame of features

        Returns:
            pd.DataFrame: DataFrame with predictions
        """
        try:
            # Keep only required columns
            required_columns = self.numerical_columns + self.categorical_columns
            input_df = df[required_columns].copy()

            # Apply preprocessing
            input_features = self.preprocessor.transform(input_df)

            # Make predictions
            predictions = self.model.predict(input_features)
            prediction_probas = self.model.predict_proba(input_features)

            # Get max probability for each prediction
            max_probas = np.max(prediction_probas, axis=1)

            # Add predictions to DataFrame
            result_df = df.copy()
            result_df['prediction'] = predictions
            result_df['probability'] = max_probas
            result_df['prediction_label'] = result_df['prediction'].map(self.target_mapping)

            return result_df

        except Exception as e:
            logging.error(f"Error making batch predictions: {str(e)}")
            raise e

    def get_feature_names(self) -> List[str]:
        """
        Get the list of feature names needed for prediction.

        Returns:
            List[str]: List of feature names
        """
        return self.numerical_columns + self.categorical_columns

    def get_feature_descriptions(self) -> Dict[str, str]:
        """
        Get descriptions for each feature.

        Returns:
            Dict[str, str]: Dictionary of feature descriptions
        """
        # Define descriptions for common heart disease features
        descriptions = {
            "age": "Age in years",
            "sex": "Sex (0 = female, 1 = male)",
            "cp": "Chest pain type (0-3)",
            "trestbps": "Resting blood pressure in mm Hg",
            "chol": "Serum cholesterol in mg/dl",
            "fbs": "Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)",
            "restecg": "Resting electrocardiographic results (0-2)",
            "thalach": "Maximum heart rate achieved",
            "exang": "Exercise induced angina (1 = yes, 0 = no)",
            "oldpeak": "ST depression induced by exercise relative to rest",
            "slope": "Slope of the peak exercise ST segment (0-2)",
            "ca": "Number of major vessels colored by flouroscopy (0-3)",
            "thal": "Thalassemia (0-3)"
        }

        return {feat: descriptions.get(feat, "No description available") for feat in self.get_feature_names()}


class ModelDeployment:
    """
    Class for deploying the model for prediction.
    """

    def __init__(self, config: ModelDeploymentConfig):
        """
        Initialize model deployment.

        Args:
            config (ModelDeploymentConfig): Configuration for model deployment
        """
        self.config = config

    def create_prediction_pipeline(self):
        """
        Create and save the prediction pipeline.

        Returns:
            PredictionPipeline: Prediction pipeline
        """
        try:
            # Create prediction pipeline
            prediction_pipeline = PredictionPipeline(
                model_path=self.config.trained_model_path,
                preprocessor_path=self.config.preprocessor_path,
                schema_file=self.config.schema_file
            )

            # Save prediction pipeline
            joblib.dump(prediction_pipeline, self.config.prediction_pipeline_path)
            logging.info(f"Prediction pipeline saved to {self.config.prediction_pipeline_path}")

            return prediction_pipeline

        except Exception as e:
            logging.error(f"Error creating prediction pipeline: {str(e)}")
            raise e

    def load_prediction_pipeline(self) -> PredictionPipeline:
        """
        Load the prediction pipeline.

        Returns:
            PredictionPipeline: Prediction pipeline
        """
        try:
            # Check if prediction pipeline exists
            if not os.path.exists(self.config.prediction_pipeline_path):
                # Create and save prediction pipeline
                return self.create_prediction_pipeline()

            # Load prediction pipeline
            prediction_pipeline = joblib.load(self.config.prediction_pipeline_path)
            logging.info(f"Prediction pipeline loaded from {self.config.prediction_pipeline_path}")

            return prediction_pipeline

        except Exception as e:
            logging.error(f"Error loading prediction pipeline: {str(e)}")
            raise e

    def create_sample_input_json(self, output_file="sample_input.json"):
        """
        Create sample input JSON for API testing.

        Args:
            output_file (str): Output file path

        Returns:
            dict: Sample input data
        """
        try:
            # Load prediction pipeline to get feature names
            pipeline = self.load_prediction_pipeline()

            # Create sample input data
            sample_input = {
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

            # Save sample input to JSON file
            import json
            with open(output_file, 'w') as f:
                json.dump(sample_input, f, indent=4)

            logging.info(f"Sample input saved to {output_file}")

            return sample_input

        except Exception as e:
            logging.error(f"Error creating sample input: {str(e)}")
            raise e