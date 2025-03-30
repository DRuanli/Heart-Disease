import os
import yaml
from box import ConfigBox
from pathlib import Path
import logging


def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """
    Reads a YAML file and returns its contents as a ConfigBox.

    Args:
        path_to_yaml (Path): Path to the YAML file

    Returns:
        ConfigBox: ConfigBox containing the YAML file contents

    Raises:
        ValueError: If the YAML file is empty
        Exception: If any other error occurs
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logging.info(f"YAML file: {path_to_yaml} loaded successfully")

            # Handle None case (empty file)
            if content is None:
                logging.warning(f"YAML file: {path_to_yaml} is empty. Using empty dict.")
                return ConfigBox({})

            return ConfigBox(content)
    except Exception as e:
        logging.error(f"Error reading YAML file: {path_to_yaml} - {str(e)}")
        raise e


def create_directories(path_to_directories: list, verbose=True):
    """
    Creates a list of directories.

    Args:
        path_to_directories (list): List of paths of directories to be created
        verbose (bool, optional): Whether to log the directory creation. Defaults to True.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logging.info(f"Created directory at: {path}")