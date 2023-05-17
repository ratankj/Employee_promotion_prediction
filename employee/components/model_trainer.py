# Basic Import
import numpy as np
import pandas as pd

from employee.exception import CustomException
from employee.logger import logging
from employee.config.configuration import MODEL_FILE_PATH
from employee.utils.utils import save_object
#from employee.utils.utils import evaluate_model
from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier



from sklearn.metrics import accuracy_score, classification_report,precision_score, recall_score, f1_score, roc_auc_score,roc_curve 



from dataclasses import dataclass
import sys
import os

@dataclass 
class ModelTrainerConfig:
    trained_model_file_path = MODEL_FILE_PATH





class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()


    def evaluate_model(self,X_train,y_train,X_test,y_test,models):
        try:
            report = {}
            
            for i in range(len(models)):
                model = list(models.values())[i]
                
                model.fit(X_train,y_train)
                
                y_test_pred = model.predict(X_test)
                
                test_model_score = accuracy_score(y_test,y_test_pred)
                
                
                report[list(models.keys())[i]] = test_model_score
            
            return report
                
        except Exception as e:
            logging.info("Exception occure while evaluation of model")
            raise CustomException(e,sys)


    def initate_model_training(self,train_array,test_array):
        try:
            logging.info('Splitting Dependent and Independent variables from train and test data')
            X_train, y_train, X_test, y_test = (train_array[:,:-1], train_array[:,-1],
                                                test_array[:,:-1],test_array[:,-1])

            models={
            'LogisticRegression':LogisticRegression(),
            'DecisionTree':DecisionTreeClassifier(),
            'Gradient Boosting':GradientBoostingClassifier(),
            'Random Forest':RandomForestClassifier(),
            'XGB Regressor':XGBClassifier(),
            'KNN neighbour':KNeighborsClassifier()
            
        }
            
           

            
            model_report:dict=self.evaluate_model(X_train,y_train,X_test,y_test,models)
          

            print(f"model_report: {model_report}")

            df = pd.DataFrame(list(model_report.items()), columns=['Model', 'Accuracy'])

            print(df)
            
            print('\n====================================================================================\n')

            logging.info(f'Model Report : {model_report}')

            # To get best model score from dictionary 
            best_model_score = max(sorted(model_report.values()))

            logging.info(f"best model score: {best_model_score}")

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            #logging.info(f"{plot_confusion_matrix(best_model_name, X_test, self.y_test_pred, cmap='Blues', values_format='d')}")

            best_model = models[best_model_name]

            print(f'Best Model Found , Model Name : {best_model_name} , Accuracy Score : {best_model_score}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found , Model Name : {best_model_name} , Accuracy Score : {best_model_score}')

            save_object(
                 file_path=self.model_trainer_config.trained_model_file_path,
                 obj=best_model
            )
          

        except Exception as e:
            logging.info('Exception occured at Model Training')
            raise CustomException(e,sys)