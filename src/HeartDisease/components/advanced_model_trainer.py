import os
import pandas as pd
import numpy as np
import json
import logging
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.utils import class_weight
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
import optuna
from src.HeartDisease.entity import ModelTrainerConfig
from src.HeartDisease.utils import read_yaml
from src.HeartDisease.constants import PARAMS_FILE_PATH


class AdvancedModelTrainer:
    """
    Advanced class for training and evaluating machine learning models with hyperparameter tuning.
    """

    def __init__(self, config: ModelTrainerConfig):
        """
        Initialize advanced model trainer.

        Args:
            config (ModelTrainerConfig): Configuration for model trainer
        """
        self.config = config
        self.params = read_yaml(PARAMS_FILE_PATH)

    def load_data(self):
        """
        Load and prepare the training and test data.

        Returns:
            tuple: (X_train, y_train, X_test, y_test)
        """
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
            logging.warning(f"Dropped {len(train_df) - len(train_df_clean)} rows with NaN in target from training set")
        if len(test_df_clean) < len(test_df):
            logging.warning(f"Dropped {len(test_df) - len(test_df_clean)} rows with NaN in target from test set")

        X_train = train_df_clean.drop(columns=[target_column])
        y_train = train_df_clean[target_column]
        X_test = test_df_clean.drop(columns=[target_column])
        y_test = test_df_clean[target_column]

        return X_train, y_train, X_test, y_test

    def handle_class_imbalance(self, X_train, y_train):
        """
        Handle class imbalance using SMOTE.

        Args:
            X_train: Training features
            y_train: Training target

        Returns:
            tuple: (X_resampled, y_resampled)
        """
        try:
            # Check class distribution
            class_counts = pd.Series(y_train).value_counts()
            logging.info(f"Class distribution before SMOTE: {class_counts.to_dict()}")

            if len(class_counts) > 1:  # SMOTE requires at least 2 classes
                # Apply SMOTE for oversampling
                smote = SMOTE(random_state=self.params.random_state)
                X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

                # Log new class distribution
                new_class_counts = pd.Series(y_resampled).value_counts()
                logging.info(f"Class distribution after SMOTE: {new_class_counts.to_dict()}")

                return X_resampled, y_resampled
            else:
                logging.warning("SMOTE not applied - insufficient unique classes")
                return X_train, y_train

        except Exception as e:
            logging.error(f"Error in SMOTE: {str(e)}")
            logging.warning("Returning original data without SMOTE")
            return X_train, y_train

    def optimize_random_forest(self, X_train, y_train, X_test, y_test, n_trials=20):
        """
        Optimize RandomForest hyperparameters using Optuna.

        Args:
            X_train: Training features
            y_train: Training target
            X_test: Test features
            y_test: Test target
            n_trials: Number of optimization trials

        Returns:
            tuple: (best_model, best_params)
        """

        def objective(trial):
            # Define hyperparameters to optimize
            n_estimators = trial.suggest_int('n_estimators', 50, 300)
            max_depth = trial.suggest_int('max_depth', 3, 30)
            min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
            min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
            max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
            bootstrap = trial.suggest_categorical('bootstrap', [True, False])
            class_weight_option = trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample', None])

            # Create model with suggested hyperparameters
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                bootstrap=bootstrap,
                class_weight=class_weight_option,
                random_state=self.params.random_state
            )

            # Use stratified k-fold cross-validation
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.params.random_state)

            # Evaluate using cross-validation
            try:
                scores = cross_val_score(model, X_train, y_train, scoring='f1_macro', cv=cv)
                return scores.mean()
            except Exception as e:
                logging.error(f"Error in cross-validation: {str(e)}")
                return 0.0  # Return low score on error

        # Create and run Optuna study
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)

        # Get best parameters
        best_params = study.best_params
        logging.info(f"Best RandomForest parameters: {best_params}")

        # Train model with best parameters
        best_model = RandomForestClassifier(
            **best_params,
            random_state=self.params.random_state
        )
        best_model.fit(X_train, y_train)

        return best_model, best_params

    def optimize_gradient_boosting(self, X_train, y_train, X_test, y_test, n_trials=20):
        """
        Optimize GradientBoosting hyperparameters using Optuna.

        Args:
            X_train: Training features
            y_train: Training target
            X_test: Test features
            y_test: Test target
            n_trials: Number of optimization trials

        Returns:
            tuple: (best_model, best_params)
        """

        def objective(trial):
            # Define hyperparameters to optimize
            n_estimators = trial.suggest_int('n_estimators', 50, 300)
            learning_rate = trial.suggest_float('learning_rate', 0.01, 0.2)
            max_depth = trial.suggest_int('max_depth', 3, 10)
            min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
            min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
            subsample = trial.suggest_float('subsample', 0.5, 1.0)

            # Create model with suggested hyperparameters
            model = GradientBoostingClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                subsample=subsample,
                random_state=self.params.random_state
            )

            # Use stratified k-fold cross-validation
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.params.random_state)

            # Evaluate using cross-validation
            try:
                scores = cross_val_score(model, X_train, y_train, scoring='f1_macro', cv=cv)
                return scores.mean()
            except Exception as e:
                logging.error(f"Error in cross-validation: {str(e)}")
                return 0.0  # Return low score on error

        # Create and run Optuna study
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)

        # Get best parameters
        best_params = study.best_params
        logging.info(f"Best GradientBoosting parameters: {best_params}")

        # Train model with best parameters
        best_model = GradientBoostingClassifier(
            **best_params,
            random_state=self.params.random_state
        )
        best_model.fit(X_train, y_train)

        return best_model, best_params

    def create_ensemble(self, X_train, y_train, X_test, y_test):
        """
        Create an ensemble model with optimized base models.

        Args:
            X_train: Training features
            y_train: Training target
            X_test: Test features
            y_test: Test target

        Returns:
            model: Trained ensemble model
        """
        # Optimize base models
        rf_model, rf_params = self.optimize_random_forest(X_train, y_train, X_test, y_test, n_trials=10)
        gb_model, gb_params = self.optimize_gradient_boosting(X_train, y_train, X_test, y_test, n_trials=10)

        # Create logistic regression model
        lr_model = LogisticRegression(
            C=1.0,
            max_iter=2000,
            multi_class='multinomial',
            solver='lbfgs',
            random_state=self.params.random_state
        )

        # Create voting ensemble
        ensemble = VotingClassifier(
            estimators=[
                ('rf', rf_model),
                ('gb', gb_model),
                ('lr', lr_model)
            ],
            voting='soft'  # Use probability estimates for voting
        )

        # Train ensemble
        ensemble.fit(X_train, y_train)

        return ensemble

    def feature_importance_analysis(self, model, X_train, feature_names=None):
        """
        Analyze feature importance from trained model.

        Args:
            model: Trained model
            X_train: Training features
            feature_names: List of feature names

        Returns:
            pd.DataFrame: DataFrame with feature importances
        """
        if not feature_names:
            feature_names = X_train.columns.tolist()

        # Get feature importances if available
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.mean(np.abs(model.coef_), axis=0)
        else:
            # For ensemble models, check if one of the base estimators has feature_importances_
            if hasattr(model, 'estimators_'):
                for est_name, estimator in model.named_estimators_.items():
                    if hasattr(estimator, 'feature_importances_'):
                        importances = estimator.feature_importances_
                        logging.info(f"Using feature importances from {est_name}")
                        break
                else:
                    logging.warning("No feature importances found in model")
                    return None
            else:
                logging.warning("No feature importances found in model")
                return None

        # Create DataFrame with feature importances
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        })
        feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)

        # Log top features
        logging.info(f"Top 10 important features: {feature_importance_df.head(10).to_dict()}")

        return feature_importance_df

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

            # Calculate cross-validation scores
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.params.random_state)
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1_macro')
            cv_f1 = cv_scores.mean()

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
            logging.info(f"Cross-validation F1: {cv_f1:.4f}")
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
                },
                "cross_validation": {
                    "f1_scores": cv_scores.tolist(),
                    "f1_mean": float(cv_f1),
                    "f1_std": float(cv_scores.std())
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

    def train(self):
        """
        Train advanced models with hyperparameter tuning and ensemble methods.

        Returns:
            tuple: (best_model, metrics)
        """
        try:
            logging.info("Starting advanced model training")

            # Load and prepare data
            X_train, y_train, X_test, y_test = self.load_data()

            # Handle class imbalance
            X_resampled, y_resampled = self.handle_class_imbalance(X_train, y_train)

            # 1. Train Random Forest with optimized hyperparameters
            logging.info("Optimizing Random Forest model")
            rf_model, rf_params = self.optimize_random_forest(X_resampled, y_resampled, X_test, y_test)
            rf_metrics = self.evaluate_model(rf_model, X_resampled, y_resampled, X_test, y_test)
            logging.info(f"Random Forest metrics: {rf_metrics}")

            # 2. Train Gradient Boosting with optimized hyperparameters
            logging.info("Optimizing Gradient Boosting model")
            gb_model, gb_params = self.optimize_gradient_boosting(X_resampled, y_resampled, X_test, y_test)
            gb_metrics = self.evaluate_model(gb_model, X_resampled, y_resampled, X_test, y_test)
            logging.info(f"Gradient Boosting metrics: {gb_metrics}")

            # 3. Create ensemble model
            logging.info("Creating ensemble model")
            ensemble_model = self.create_ensemble(X_resampled, y_resampled, X_test, y_test)
            ensemble_metrics = self.evaluate_model(ensemble_model, X_resampled, y_resampled, X_test, y_test)
            logging.info(f"Ensemble metrics: {ensemble_metrics}")

            # Compare models and select the best
            models_with_metrics = {
                "RandomForest": (rf_model, rf_metrics, rf_params),
                "GradientBoosting": (gb_model, gb_metrics, gb_params),
                "Ensemble": (ensemble_model, ensemble_metrics, None)
            }

            # Select best model based on test F1 score
            best_model_name = max(
                models_with_metrics.keys(),
                key=lambda k: models_with_metrics[k][1]["test"]["f1"]
            )
            best_model, best_metrics, best_params = models_with_metrics[best_model_name]

            logging.info(f"Best model: {best_model_name}")

            # Analyze feature importance for best model
            feature_importance = self.feature_importance_analysis(best_model, X_train, X_train.columns)

            # Add feature importance to metrics if available
            if feature_importance is not None:
                top_features = feature_importance.head(10).to_dict(orient='records')
                best_metrics["feature_importance"] = top_features

            # Add best model parameters to metrics
            if best_params:
                best_metrics["best_params"] = best_params

            # Save best model
            joblib.dump(best_model, self.config.trained_model_path)
            logging.info(f"Best model saved to {self.config.trained_model_path}")

            # Save metrics
            with open(self.config.metrics_path, 'w') as f:
                json.dump(best_metrics, f, indent=4)
            logging.info(f"Metrics saved to {self.config.metrics_path}")

            return best_model, best_metrics

        except Exception as e:
            logging.error(f"Error in advanced model training: {str(e)}")
            raise e