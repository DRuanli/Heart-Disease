import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import logging
import joblib
from src.HeartDisease.entity import DataTransformationConfig
from src.HeartDisease.utils import read_yaml
from src.HeartDisease.constants import PARAMS_FILE_PATH

class DataTransformation:
    """
    Class responsible for transforming the data.
    """

    def __init__(self, config: DataTransformationConfig):
        """
        Initialize data transformation.

        Args:
            config (DataTransformationConfig): Configuration for data transformation
        """
        self.config = config
        self.schema = read_yaml(Path("config/schema.yaml"))
        self.params = read_yaml(PARAMS_FILE_PATH)

    def get_data(self):
        """
        Read and combine the processed data files.

        Returns:
            pd.DataFrame: Combined dataset
        """
        try:
            # For Heart Disease, we'll focus on the processed data files which are more consistent
            processed_files = [
                "processed.cleveland.data",
                "processed.hungarian.data",
                "processed.switzerland.data",
                "processed.va.data"
            ]

            column_names = list(self.schema.columns.keys())
            all_data = []

            for file in processed_files:
                file_path = os.path.join(self.config.data_dir, file)
                try:
                    # Try multiple encodings as UCI data can be inconsistent
                    for encoding in ['utf-8', 'latin-1', 'ISO-8859-1']:
                        try:
                            df = pd.read_csv(file_path, header=None, sep=',',
                                             na_values='?', names=column_names,
                                             encoding=encoding)
                            all_data.append(df)
                            logging.info(f"Successfully read {file} with {encoding} encoding")
                            break
                        except Exception as e:
                            logging.warning(f"Failed to read {file} with {encoding} encoding: {str(e)}")
                            continue
                except Exception as e:
                    logging.error(f"Error reading {file}: {str(e)}")
                    continue

            combined_data = pd.concat(all_data, ignore_index=True)
            logging.info(f"Combined {len(all_data)} datasets with total {len(combined_data)} records")
            return combined_data

        except Exception as e:
            logging.error(f"Error in get_data: {str(e)}")
            raise e

    def handle_missing_values(self, df):
        """
        Handle missing values in the dataset.

        Args:
            df (pd.DataFrame): Input DataFrame

        Returns:
            pd.DataFrame: DataFrame with missing values handled
        """
        logging.info("Handling missing values")
        # Log missing values count before imputation
        missing_values = df.isnull().sum()
        logging.info(f"Missing values before imputation:\n{missing_values[missing_values > 0]}")

        # Save the DataFrame with missing values handled
        return df

    def create_preprocessor(self):
        """
        Create a scikit-learn preprocessor for numerical and categorical features.

        Returns:
            ColumnTransformer: Scikit-learn preprocessor
        """
        logging.info("Creating preprocessor")

        # Get numerical and categorical columns from schema
        numerical_features = self.schema.numerical_columns
        categorical_features = self.schema.categorical_columns

        # Numerical pipeline with median imputation and standard scaling
        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        # Categorical pipeline with most frequent imputation and one-hot encoding
        cat_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
        ])

        # Combine both pipelines
        preprocessor = ColumnTransformer([
            ('num_pipeline', num_pipeline, numerical_features),
            ('cat_pipeline', cat_pipeline, categorical_features)
        ])

        return preprocessor

    def transform_data(self):
        """
        Transform the data and save the preprocessor and split datasets.
        """
        try:
            logging.info("Starting data transformation")

            # Get the data
            df = self.get_data()
            logging.info(f"Dataset shape: {df.shape}")

            # Handle missing values
            df = self.handle_missing_values(df)

            # Separate features and target
            target_column = self.schema.target_column
            X = df.drop(columns=[target_column])
            y = df[target_column]

            # Create preprocessor
            preprocessor = self.create_preprocessor()

            # Fit and transform the data
            X_transformed = preprocessor.fit_transform(X)

            # Create column names for transformed data
            feature_names = self.get_feature_names(preprocessor, X.columns)

            # Convert to DataFrame with feature names
            X_transformed_df = pd.DataFrame(X_transformed, columns=feature_names)

            # Add target column back
            transformed_df = pd.concat([X_transformed_df, pd.Series(y, name=target_column)], axis=1)

            # Save preprocessed data
            transformed_df.to_csv(self.config.preprocessed_file, index=False)

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_transformed, y,
                test_size=self.params.model_params.test_size,
                random_state=self.params.random_state
            )

            # Create DataFrames for train and test sets
            train_df = pd.DataFrame(X_train, columns=feature_names)
            train_df[target_column] = y_train

            test_df = pd.DataFrame(X_test, columns=feature_names)
            test_df[target_column] = y_test

            # Save train and test sets
            train_df.to_csv(self.config.train_data_path, index=False)
            test_df.to_csv(self.config.test_data_path, index=False)

            # Save preprocessor
            joblib.dump(preprocessor, self.config.preprocessor_path)

            logging.info("Data transformation completed")
            logging.info(f"Train dataset shape: {train_df.shape}, Test dataset shape: {test_df.shape}")

            return self.config.preprocessed_file, self.config.train_data_path, self.config.test_data_path

        except Exception as e:
            logging.error(f"Error in transform_data: {str(e)}")
            raise e

    def get_feature_names(self, column_transformer, input_features):
        """
        Get feature names from column transformer.

        Args:
            column_transformer: ColumnTransformer object
            input_features: Input feature names

        Returns:
            list: List of transformed feature names
        """
        output_features = []

        for name, pipe, features in column_transformer.transformers_:
            if name == 'num_pipeline':
                output_features.extend(features)
            elif name == 'cat_pipeline':
                for i, feature in enumerate(features):
                    cats = pipe.named_steps['encoder'].categories_[i]
                    # Skip the first category for each feature (drop='first')
                    output_features.extend([f"{feature}_{cat}" for cat in cats[1:]])

        return output_features