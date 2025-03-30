from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class DataIngestionConfig:
    """
    Configuration for data ingestion.
    """
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path


@dataclass(frozen=True)
class DataValidationConfig:
    """
    Configuration for data validation.
    """
    root_dir: Path
    data_dir: Path
    status_file: Path
    required_files: List[str]
    schema_file: Path


@dataclass(frozen=True)
class DataTransformationConfig:
    """
    Configuration for data transformation.
    """
    root_dir: Path
    data_dir: Path
    processed_data_dir: Path
    preprocessed_file: Path
    train_data_path: Path
    test_data_path: Path
    preprocessor_path: Path