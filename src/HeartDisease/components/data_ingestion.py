import os
import urllib.request as request
import zipfile
from src.HeartDisease.entity import DataIngestionConfig
from src.HeartDisease.utils import create_directories
import gdown
import logging


class DataIngestion:
    """
    Class for data ingestion operations.
    """

    def __init__(self, config: DataIngestionConfig):
        """
        Initialize data ingestion.

        Args:
            config (DataIngestionConfig): Configuration for data ingestion
        """
        self.config = config

    def download_file(self):
        """
        Download file from source URL to local data file.
        """
        if not os.path.exists(self.config.local_data_file):
            # Extract file ID from Google Drive URL
            file_id = self.config.source_URL.split("/")[-2]
            prefix = "https://drive.google.com/uc?id="
            download_url = f"{prefix}{file_id}"

            logging.info(f"Downloading data from {self.config.source_URL}")
            # Convert Path object to string for gdown
            gdown.download(download_url, str(self.config.local_data_file), quiet=False)
            logging.info(f"Downloaded data to {self.config.local_data_file}")
        else:
            logging.info(f"File already exists at {self.config.local_data_file}")

    def extract_zip_file(self):
        """
        Extract the zip file to the unzip directory.
        """
        create_directories([self.config.unzip_dir])

        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(self.config.unzip_dir)

        logging.info(f"Extracted zip file to {self.config.unzip_dir}")