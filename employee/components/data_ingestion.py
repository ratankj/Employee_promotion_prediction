from employee.exception import CustomException
from employee.logger import logging
from employee.constant import *
from employee.config.configuration import *
import os,sys
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from employee.components.data_transformation import DataTransformation
from employee.components import data_transformation
from dataclasses import dataclass
from employee.components.model_trainer import ModelTrainer


@dataclass
class DataIngestionconfig:
    
    train_data_path:str= TRAIN_FILE_PATH
    test_data_path:str= TEST_FILE_PATH
    raw_data_path:str=RAW_FILE_PATH

class DataIngestion:
    def __init__(self):
        self.data_ingestion_config=DataIngestionconfig()


    def initiate_data_ingestion(self):
        logging.info(" data ingestion started...")
        logging.info(f"data set path: {DATASET_PATH}")

        try:
            df=pd.read_csv(DATASET_PATH)

            logging.info("tring to add")
            logging.info(f"dataset columns {df.columns}")

            employee_file_name=os.path.basename(DATASET_PATH)
            

            logging.info(f"basename: {employee_file_name}")


            os.makedirs(os.path.dirname(self.data_ingestion_config.raw_data_path),exist_ok=True)


            
            df.to_csv(self.data_ingestion_config.raw_data_path,index=False)
            

            df.drop(['employee_id'],axis=1,inplace=True)
            df.rename(columns = {"awards_won?": "award_won","KPIs_met >80%":"kpi_80"}, inplace = True) 
            logging.info(" ***********rename column*************")

            logging.info(f"dataset columns {df.columns}")



            

            logging.info("train test split")

            train_set,test_Set = train_test_split(df,test_size=0.20,random_state=42)

            os.makedirs(os.path.dirname(self.data_ingestion_config.train_data_path),exist_ok=True)
            train_set.to_csv(self.data_ingestion_config.train_data_path,index=False,header=True)
           

            logging.info(f"train data path, {TRAIN_FILE_PATH}")

            os.makedirs(os.path.dirname(self.data_ingestion_config.test_data_path),exist_ok=True)
            test_Set.to_csv(self.data_ingestion_config.test_data_path,index=False,header=True)
            
            
            
            logging.info(f"test data path, {TEST_FILE_PATH}")

            logging.info("data ingestion complete")


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
    data_transformation=DataTransformation()
    train_arr,test_arr,_= data_transformation.initaite_data_transformation(train_data,test_data)
    model_trainer=ModelTrainer()
    model_trainer.initate_model_training(train_arr,test_arr)




# to run data ingestion

# python employee/components/data_ingestion.py

    
