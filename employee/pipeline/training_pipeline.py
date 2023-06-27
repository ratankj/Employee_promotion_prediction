import os
import sys
from employee.logger import logging
from employee.exception import CustomException
import pandas as pd

from employee.components.data_ingestion import DataIngestion
from employee.components.data_transformation import DataTransformation
from employee.components.model_trainer import ModelTrainer

class Train:
    def __init__(self):
        pass

    def main(self):

        if __name__=='__main__':
            obj=DataIngestion()
            train_data_path,test_data_path=obj.initiate_data_ingestion()
            data_transformation = DataTransformation()
            train_arr,test_arr,_=data_transformation.initaite_data_transformation(train_data_path,test_data_path)
            model_trainer=ModelTrainer()
            model_trainer.initate_model_training(train_arr,test_arr)

