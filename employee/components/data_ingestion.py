from employee.exception import CustomException
from employee.logger import logging
from employee.constant import *
from employee.config.configuration import *
import os,sys
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from dataclasses import dataclass


@dataclass
class DataIngestionconfig:
    raw_data_path:str=RAW_FILE_PATH
    train_data_path:str= TRAIN_FILE_PATH
    test_data_path:str= TEST_FILE_PATH

class DataIngestion:
    def __init__(self):
        self.data_ingestion_config=DataIngestionconfig()


    def initiate_data_ingestion(self):
        logging.info(" data ingestion started...")

        try:
            df=pd.read_csv(DATASET_PATH)
        
            #logging.info(f"dataset path : {DATASET_PATH}"

            os.makedirs(os.path.dirname(self.data_ingestion_config.raw_data_path),exist_ok=True)
            df.to_csv(self.data_ingestion_config.raw_data_path,index=False)
            

            logging.info("train test split")

            train_set,test_Set = train_test_split(df,test_size=0.20,random_state=42)

            os.makedirs(os.path.dirname(self.data_ingestion_config.train_data_path),exist_ok=True)
            train_set.to_csv(self.data_ingestion_config.train_data_path,index=False,header=True)

            os.makedirs(os.path.dirname(self.data_ingestion_config.test_data_path),exist_ok=True)
            test_Set.to_csv(self.data_ingestion_config.test_data_path,index=False,header=True)
            
            logging.info(f"train data path, {TRAIN_FILE_PATH}")

            logging.info("data insgested into raw, train and test data")

            return(

                self.data_ingestion_config.train_data_path,
                self.data_ingestion_config.test_data_path
            )


        except Exception as e:
            logging.info("Exception in data ingestion stage")
            raise CustomException(e,sys)



    
# to run data ingestion

if __name__ == "__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()

    
