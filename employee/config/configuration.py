import os,sys
from employee.constant import *
from employee.exception import CustomException
from employee.logger import logging

ROOT_DIR=ROOT_DIR_KEY
DATASET_PATH=os.path.join(ROOT_DIR,DATA_DIR,DATA_DIR_KEY)


RAW_FILE_PATH=os.path.join(ROOT_DIR,ARTIFACT_DIR_KEY,DATA_INGESTION_KEY,
                           CURRENT_TIME_STAMP,DATA_INGESTION_RAW_DATA_DIR_KEY,RAW_DATA_DIR_KEY)


TRAIN_FILE_PATH=os.path.join(ROOT_DIR,ARTIFACT_DIR_KEY,DATA_INGESTION_KEY,
                           CURRENT_TIME_STAMP,DATA_INGESTION_INGESTED_DIR_NAME_KEY,TRAIN_DATA_DIR_KEY)

TEST_FILE_PATH=os.path.join(ROOT_DIR,ARTIFACT_DIR_KEY,DATA_INGESTION_KEY,
                           CURRENT_TIME_STAMP,DATA_INGESTION_INGESTED_DIR_NAME_KEY,TEST_DATA_DIR_KEY)



# validation

VALIDATION_TRAIN_FILE_PATH=os.path.join(ROOT_DIR,ARTIFACT_DIR_KEY,DATA_VALIDATION_KEY,
                                  CURRENT_TIME_STAMP,DATA_VALIDATION_VALIDATED_DIR_KEY,VALIDATED_TRAIN_DIR_KEY)



VALIDATION_TEST_FILE_PATH=os.path.join(ROOT_DIR,ARTIFACT_DIR_KEY,DATA_VALIDATION_KEY,
                                  CURRENT_TIME_STAMP,DATA_VALIDATION_VALIDATED_DIR_KEY,VALIDATED_TEST_DIR_KEY)





# PREPROCESSING OBJ


PREPROCESSING_OBJ_PATH = os.path.join(ROOT_DIR,ARTIFACT_DIR_KEY,
                                      DATA_TRANSFORMATION_ARTIFACT,CURRENT_TIME_STAMP,DATA_PREPROCESSED_DIR,
                                      DATA_TRANSFORMATION_PREPROCESSING_OBJ)

# MODEL FILE

MODEL_FILE_PATH = os.path.join(ROOT_DIR,ARTIFACT_DIR_KEY,
                               MODEL_ARTIFACT,CURRENT_TIME_STAMP,MODEL_OBJECT)

