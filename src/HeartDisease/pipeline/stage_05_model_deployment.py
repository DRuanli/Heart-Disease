from src.HeartDisease.config.configuration import ConfigurationManager
from src.HeartDisease.components.model_deployment import ModelDeployment
import logging


def main():
    """
    Main function to run the model deployment pipeline.
    """
    try:
        config = ConfigurationManager()
        model_deployment_config = config.get_model_deployment_config()
        model_deployment = ModelDeployment(model_deployment_config)

        # Create and save prediction pipeline
        pipeline = model_deployment.create_prediction_pipeline()

        # Create sample input JSON for API testing
        sample_input = model_deployment.create_sample_input_json()

        logging.info("Model deployment completed successfully")

        return pipeline

    except Exception as e:
        logging.error(f"Error in model deployment pipeline: {str(e)}")
        raise e


if __name__ == "__main__":
    main()