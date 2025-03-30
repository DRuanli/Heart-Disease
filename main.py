import logging
from src.HeartDisease.pipeline.stage_01_data_ingestion import main as data_ingestion_main
from src.HeartDisease.pipeline.stage_02_data_validation import main as data_validation_main
from src.HeartDisease.pipeline.stage_03_data_transformation import main as data_transformation_main

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
        #data_ingestion_main()
        logging.info(">>>>> Stage 1: Data Ingestion Completed <<<<<\n")

        logging.info("\n********************")
        logging.info(">>>>> Stage 2: Data Validation Started <<<<<")
        #data_validation_main()
        logging.info(">>>>> Stage 2: Data Validation Completed <<<<<\n")

        logging.info("\n********************")
        logging.info(">>>>> Stage 3: Data Transformation Started <<<<<")
        data_transformation_main()
        logging.info(">>>>> Stage 3: Data Transformation Completed <<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e


if __name__ == "__main__":
    main()