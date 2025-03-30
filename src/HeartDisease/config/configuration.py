from src.HeartDisease.constants import *
from src.HeartDisease.utils import read_yaml, create_directories
from src.HeartDisease.entity import DataIngestionConfig, DataValidationConfig
from pathlib import Path


class ConfigurationManager:
    """
    Class to manage configurations for the project.
    """

    def __init__(
            self,
            config_filepath=CONFIG_FILE_PATH,
            params_filepath=PARAMS_FILE_PATH):
        """
        Initialize the configuration manager.

        Args:
            config_filepath (Path, optional): Path to the config YAML. Defaults to CONFIG_FILE_PATH.
            params_filepath (Path, optional): Path to the params YAML. Defaults to PARAMS_FILE_PATH.
        """
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        """
        Get data ingestion configuration.

        Returns:
            DataIngestionConfig: Data ingestion configuration
        """
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=Path(config.root_dir),
            source_URL=config.source_URL,
            local_data_file=Path(config.local_data_file),
            unzip_dir=Path(config.unzip_dir)
        )

        return data_ingestion_config

    def get_data_validation_config(self) -> DataValidationConfig:
        """
        Get data validation configuration.

        Returns:
            DataValidationConfig: Data validation configuration
        """
        config = self.config.data_validation

        create_directories([config.root_dir])

        data_validation_config = DataValidationConfig(
            root_dir=Path(config.root_dir),
            data_dir=Path(config.data_dir),
            status_file=Path(config.status_file),
            required_files=config.required_files,
            schema_file=Path(config.schema_file)
        )

        return data_validation_config