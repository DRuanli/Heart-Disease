from src.HeartDisease.config.configuration import ConfigurationManager
from src.HeartDisease.components.advanced_model_trainer import AdvancedModelTrainer
import logging


def main():
    """
    Main function to run the advanced model training pipeline.
    """
    try:
        config = ConfigurationManager()
        model_trainer_config = config.get_model_trainer_config()
        advanced_trainer = AdvancedModelTrainer(model_trainer_config)

        # Train advanced models with hyperparameter tuning
        best_model, best_metrics = advanced_trainer.train()

        logging.info("Advanced model training completed successfully")

        # Print a summary of the best model's performance
        if best_metrics and "test" in best_metrics:
            test_metrics = best_metrics["test"]
            logging.info(f"Best model performance summary:")
            logging.info(f"  Test Accuracy: {test_metrics.get('accuracy', 'N/A')}")
            logging.info(f"  Test F1 Score: {test_metrics.get('f1', 'N/A')}")
            logging.info(f"  Test Precision: {test_metrics.get('precision', 'N/A')}")
            logging.info(f"  Test Recall: {test_metrics.get('recall', 'N/A')}")
            if "roc_auc" in test_metrics:
                logging.info(f"  Test ROC AUC: {test_metrics.get('roc_auc', 'N/A')}")

        return best_model, best_metrics

    except Exception as e:
        logging.error(f"Error in advanced model training pipeline: {str(e)}")
        raise e


if __name__ == "__main__":
    main()