from src.HeartDisease.config.configuration import ConfigurationManager
from src.HeartDisease.components.data_validation import DataValidation
import logging


def main():
    """
    Main function to run the data validation pipeline.
    """
    config = ConfigurationManager()
    data_validation_config = config.get_data_validation_config()
    data_validation = DataValidation(data_validation_config)
    validation_status = data_validation.validate()

    if not validation_status:
        raise Exception("Data validation failed. See logs for details.")

    return validation_status


if __name__ == "__main__":
    main()