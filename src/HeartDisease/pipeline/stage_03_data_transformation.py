from src.HeartDisease.config.configuration import ConfigurationManager
from src.HeartDisease.components.data_transformation import DataTransformation
import logging


def main():
    """
    Main function to run data transformation pipeline.
    """
    try:
        config = ConfigurationManager()
        data_transformation_config = config.get_data_transformation_config()
        data_transformation = DataTransformation(data_transformation_config)
        preprocessed_file, train_data_path, test_data_path = data_transformation.transform_data()

        logging.info(f"Preprocessed data saved to: {preprocessed_file}")
        logging.info(f"Train data saved to: {train_data_path}")
        logging.info(f"Test data saved to: {test_data_path}")

        return preprocessed_file, train_data_path, test_data_path

    except Exception as e:
        logging.error(f"Error in data transformation pipeline: {str(e)}")
        raise e


if __name__ == "__main__":
    main()