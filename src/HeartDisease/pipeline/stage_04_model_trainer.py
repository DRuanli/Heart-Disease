from src.HeartDisease.config.configuration import ConfigurationManager
from src.HeartDisease.components.model_trainer import ModelTrainer
import logging

def main():
    """
    Main function to run the model training pipeline.
    """
    try:
        config = ConfigurationManager()
        model_trainer_config = config.get_model_trainer_config()
        model_trainer = ModelTrainer(model_trainer_config)

        # Option 1: Train a specific model
        # model, metrics = model_trainer.train(model_name="RandomForest")

        # Option 2: Compare multiple models and choose the best
        best_model, best_metrics = model_trainer.compare_models()

        logging.info("Model training completed successfully")
        return best_model, best_metrics

    except Exception as e:
        logging.error(f"Error in model training pipeline: {str(e)}")
        raise e

if __name__ == "__main__":
    main()