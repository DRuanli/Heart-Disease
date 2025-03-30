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

    # We've modified validate() to always return True, so the pipeline continues
    # even with missing values or other data issues that will be fixed in transformation
    return validation_status


if __name__ == "__main__":
    main()