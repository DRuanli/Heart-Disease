import os
import pandas as pd
import numpy as np
import json
import logging
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from src.HeartDisease.entity import ModelTrainerConfig
from src.HeartDisease.utils import read_yaml
from src.HeartDisease.constants import PARAMS_FILE_PATH


class ModelTrainer:
    """
    Class for training and evaluating machine learning models.
    """

    def __init__(self, config: ModelTrainerConfig):
        """
        Initialize model trainer.

        Args:
            config (ModelTrainerConfig): Configuration for model trainer
        """
        self.config = config
        self.params = read_yaml(PARAMS_FILE_PATH)

    def get_model(self, model_name="RandomForest"):
        """
        Get a model instance with configured parameters.

        Args:
            model_name (str, optional): Name of the model. Defaults to "RandomForest".

        Returns:
            model: A model instance
        """
        try:
            model_params = self.params.model_params.models[model_name]

            if model_name == "RandomForest":
                model = RandomForestClassifier(
                    n_estimators=model_params.n_estimators,
                    max_depth=model_params.max_depth,
                    min_samples_split=model_params.min_samples_split,
                    min_samples_leaf=model_params.min_samples_leaf,
                    random_state=self.params.random_state
                )
            elif model_name == "LogisticRegression":
                model = LogisticRegression(
                    C=model_params.C,
                    max_iter=model_params.max_iter,
                    penalty=model_params.penalty,
                    random_state=self.params.random_state
                )
            elif model_name == "SVC":
                model = SVC(
                    C=model_params.C,
                    kernel=model_params.kernel,
                    probability=model_params.probability,
                    random_state=self.params.random_state
                )
            else:
                raise ValueError(f"Model {model_name} not supported")

            return model

        except Exception as e:
            logging.error(f"Error getting model {model_name}: {str(e)}")
            raise e

    def train(self, model_name="RandomForest"):
        """
        Train a model on the training data.

        Args:
            model_name (str, optional): Name of the model. Defaults to "RandomForest".

        Returns:
            trained_model: Trained model
        """
        try:
            logging.info(f"Starting model training with {model_name}")

            # Load training data
            train_data_path = "artifacts/data_transformation/train_test_split/train.csv"
            test_data_path = "artifacts/data_transformation/train_test_split/test.csv"

            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)

            logging.info(f"Loaded training data: {train_df.shape} and test data: {test_df.shape}")

            # Split into features and target
            target_column = self.params.target_column

            # Handle NaN values in the target column by dropping those rows
            train_df_clean = train_df.dropna(subset=[target_column])
            test_df_clean = test_df.dropna(subset=[target_column])

            if len(train_df_clean) < len(train_df):
                logging.warning(
                    f"Dropped {len(train_df) - len(train_df_clean)} rows with NaN in target from training set")
            if len(test_df_clean) < len(test_df):
                logging.warning(f"Dropped {len(test_df) - len(test_df_clean)} rows with NaN in target from test set")

            X_train = train_df_clean.drop(columns=[target_column])
            y_train = train_df_clean[target_column]
            X_test = test_df_clean.drop(columns=[target_column])
            y_test = test_df_clean[target_column]

            # Get model
            model = self.get_model(model_name)

            # Train model
            model.fit(X_train, y_train)
            logging.info(f"Model {model_name} trained successfully")

            # Evaluate
            metrics = self.evaluate_model(model, X_train, y_train, X_test, y_test)

            # Save model
            joblib.dump(model, self.config.trained_model_path)
            logging.info(f"Model saved to {self.config.trained_model_path}")

            # Save metrics
            with open(self.config.metrics_path, 'w') as f:
                json.dump(metrics, f, indent=4)
            logging.info(f"Metrics saved to {self.config.metrics_path}")

            return model, metrics

        except Exception as e:
            logging.error(f"Error training model: {str(e)}")
            raise e

    def evaluate_model(self, model, X_train, y_train, X_test, y_test):
        """
        Evaluate the model on training and test data.

        Args:
            model: Trained model
            X_train: Training features
            y_train: Training target
            X_test: Test features
            y_test: Test target

        Returns:
            dict: Dictionary of evaluation metrics
        """
        try:
            # Make predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Check if binary or multi-class classification
            n_classes = len(np.unique(y_train))
            average_method = 'binary' if n_classes == 2 else 'macro'

            logging.info(f"Target has {n_classes} classes. Using '{average_method}' averaging for metrics.")

            # Calculate metrics with appropriate averaging
            train_accuracy = accuracy_score(y_train, y_train_pred)
            test_accuracy = accuracy_score(y_test, y_test_pred)

            train_precision = precision_score(y_train, y_train_pred, average=average_method, zero_division=0)
            test_precision = precision_score(y_test, y_test_pred, average=average_method, zero_division=0)

            train_recall = recall_score(y_train, y_train_pred, average=average_method, zero_division=0)
            test_recall = recall_score(y_test, y_test_pred, average=average_method, zero_division=0)

            train_f1 = f1_score(y_train, y_train_pred, average=average_method, zero_division=0)
            test_f1 = f1_score(y_test, y_test_pred, average=average_method, zero_division=0)

            # For ROC AUC, need probability estimates
            train_roc_auc = None
            test_roc_auc = None

            if hasattr(model, "predict_proba"):
                if n_classes == 2:
                    # Binary classification
                    y_train_proba = model.predict_proba(X_train)[:, 1]
                    y_test_proba = model.predict_proba(X_test)[:, 1]
                    train_roc_auc = roc_auc_score(y_train, y_train_proba)
                    test_roc_auc = roc_auc_score(y_test, y_test_proba)
                else:
                    # Multi-class with OVR strategy
                    y_train_proba = model.predict_proba(X_train)
                    y_test_proba = model.predict_proba(X_test)
                    train_roc_auc = roc_auc_score(y_train, y_train_proba, multi_class='ovr', average='macro')
                    test_roc_auc = roc_auc_score(y_test, y_test_proba, multi_class='ovr', average='macro')

            # Get confusion matrix
            test_cm = confusion_matrix(y_test, y_test_pred)

            # Log metrics
            logging.info(f"Training accuracy: {train_accuracy:.4f}, Test accuracy: {test_accuracy:.4f}")
            logging.info(f"Training precision: {train_precision:.4f}, Test precision: {test_precision:.4f}")
            logging.info(f"Training recall: {train_recall:.4f}, Test recall: {test_recall:.4f}")
            logging.info(f"Training F1: {train_f1:.4f}, Test F1: {test_f1:.4f}")
            if train_roc_auc and test_roc_auc:
                logging.info(f"Training ROC AUC: {train_roc_auc:.4f}, Test ROC AUC: {test_roc_auc:.4f}")

            # Create metrics dictionary
            metrics = {
                "train": {
                    "accuracy": float(train_accuracy),
                    "precision": float(train_precision),
                    "recall": float(train_recall),
                    "f1": float(train_f1)
                },
                "test": {
                    "accuracy": float(test_accuracy),
                    "precision": float(test_precision),
                    "recall": float(test_recall),
                    "f1": float(test_f1),
                    "confusion_matrix": test_cm.tolist()
                }
            }

            # Add ROC AUC if available
            if train_roc_auc and test_roc_auc:
                metrics["train"]["roc_auc"] = float(train_roc_auc)
                metrics["test"]["roc_auc"] = float(test_roc_auc)

            return metrics

        except Exception as e:
            logging.error(f"Error evaluating model: {str(e)}")

            # Instead of failing, return basic metrics
            try:
                # Try to at least get accuracy which works for multi-class
                train_accuracy = accuracy_score(y_train, y_train_pred)
                test_accuracy = accuracy_score(y_test, y_test_pred)

                metrics = {
                    "train": {"accuracy": float(train_accuracy)},
                    "test": {"accuracy": float(test_accuracy)}
                }

                logging.info(
                    f"Fallback metrics - Training accuracy: {train_accuracy:.4f}, Test accuracy: {test_accuracy:.4f}")
                return metrics
            except:
                # If even that fails, return empty metrics
                return {"train": {}, "test": {}}

    def compare_models(self):
        """
        Train and compare multiple models.

        Returns:
            best_model: The best model based on test F1 score
        """
        try:
            model_names = list(self.params.model_params.models.keys())
            logging.info(f"Comparing models: {model_names}")

            # Train and evaluate each model
            results = {}
            best_score = 0
            best_model_name = None

            for model_name in model_names:
                logging.info(f"Training model: {model_name}")
                try:
                    model, metrics = self.train(model_name)

                    # Compare based on test F1 score or accuracy if F1 not available
                    if "f1" in metrics["test"]:
                        test_score = metrics["test"]["f1"]
                    else:
                        test_score = metrics["test"]["accuracy"]

                    results[model_name] = test_score

                    if test_score > best_score:
                        best_score = test_score
                        best_model_name = model_name

                except Exception as e:
                    logging.error(f"Error training {model_name}: {str(e)}")
                    continue

            # Log comparison results
            logging.info(f"Model comparison results:")
            for model_name, score in results.items():
                logging.info(f"{model_name}: Test score = {score:.4f}")

            if best_model_name:
                logging.info(f"Best model: {best_model_name} with Test score = {best_score:.4f}")

                # Train the best model again to save it
                best_model, metrics = self.train(best_model_name)
                return best_model, metrics
            else:
                logging.warning("No models were successfully trained")
                return None, None

        except Exception as e:
            logging.error(f"Error comparing models: {str(e)}")
            raise e