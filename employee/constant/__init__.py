import os,sys
from datetime import datetime

# to get the time stamp

def get_time_stamp():
    return f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

CURRENT_TIME = get_time_stamp()

ROOT_DIR_KEY =  os.getcwd()

DATA_DIR = "data"
DATA_DIR_KEY = "employee_pro.csv"

# artifact

ARTIFACTS_DIR = "Artifact"

 #Data Ingestion constants

DATA_INGESTION_KEY = 'data_ingestion'
DATA_INGESTION_RAW_DATA_DIR_KEY= "raw_data_dir"
DATA_INGESTION_INGESTED_DIR_NAME_KEY= "ingested_dir"
RAW_DATA_DIR_KEY = 'raw.csv'
TRAIN_DATA_DIR_KEY = 'train.csv'
TEST_DATA_DIR_KEY = 'test.csv' 


# data validation

DATA_VALIDATION_KEY = 'data_validation'

DATA_VALIDATION_VALIDATED_DIR_KEY = 'validated_dir'

# data validation constants
DATA_TRANSFORMATION_ARTIFACT = 'data_transformation'
DATA_TRANSFORMATION_PREPROCESSING_OBJ = 'preprocessor.pkl'



# model trainer constants
MODEL_ARTIFACT = 'model_trainer'
MODEL_OBJECT = 'model.pkl'


