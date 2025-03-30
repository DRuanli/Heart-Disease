from src.HeartDisease.config.configuration import ConfigurationManager
from src.HeartDisease.components.data_ingestion import DataIngestion
import logging

def main():
    """
    Main function to run data ingestion pipeline.
    """
    config = ConfigurationManager()
    data_ingestion_config = config.get_data_ingestion_config()
    data_ingestion = DataIngestion(data_ingestion_config)
    data_ingestion.download_file()
    data_ingestion.extract_zip_file()

if __name__ == "__main__":
    main()