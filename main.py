import logging
from src.HeartDisease.pipeline.stage_01_data_ingestion import main as data_ingestion_main

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s: %(levelname)s]: %(message)s"
)

def main():
    """
    Main function to run all pipelines.
    """
    try:
        logging.info("\n********************")
        logging.info(">>>>> Stage 1: Data Ingestion Started <<<<<")
        data_ingestion_main()
        logging.info(">>>>> Stage 1: Data Ingestion Completed <<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e

if __name__ == "__main__":
    main()