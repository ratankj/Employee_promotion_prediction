import os,sys
import pandas as pd
from employee.exception import CustomException
from employee.logger import logging
from employee.utils.utils import load_model
from employee.config.configuration import  PREPROCESSING_OBJ_PATH,MODEL_FILE_PATH

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            preprocessor_path=PREPROCESSING_OBJ_PATH
            #preprocessor_path='D:\ALL_PROJECT FOLDER\EMPLOYEE_PROMOTION\Employee_promotion_prediction\Artifact\data_transformation\2023-05-19 08-43-40\preprocessed\preprocessor.pkl'
            model_path = MODEL_FILE_PATH
            #model_path='D:\ALL_PROJECT FOLDER\EMPLOYEE_PROMOTION\Employee_promotion_prediction\Artifact\model_trainer\2023-05-19 08-43-40\model.pkl'
            
            preprocessor = load_model(preprocessor_path)
            model = load_model(model_path)


            data_scaled = preprocessor.transform(features)

            pred = model.predict(data_scaled)

            return pred
        


        except Exception as e:
            logging.info("Could not predict")
            raise CustomException(e,sys)
        

class CustomData:
    def __init__(self,
                 department:str,
                 education:str,
                 gender:str,
                 no_of_trainings:int,
                 age:int,
                 previous_year_rating:float,
                 length_of_service:int,
                 kpi_80:int,
                 award_won:int,
                 avg_training_score:int,
                 sum_metric:float,
                 total_score:int
                 ):
        
        self.department=department
        self.education=education
        self.gender=gender
        self.no_of_trainings=no_of_trainings
        self.age=age
        self.previous_year_rating=previous_year_rating
        self.length_of_service=length_of_service
        self.kpi_80=kpi_80
        self.award_won=award_won
        self.avg_training_score=avg_training_score
        self.sum_metric=sum_metric
        self.total_score=total_score


    def get_data_as_dataframe(self):
        try:
            custom_Data_input_dict={
                'department':[self.department],
                'education':[self.education],
                'gender':[self.gender],
                'no_of_trainings':[self.no_of_trainings],
                'age':[self.age],
                'previous_year_rating':[self.previous_year_rating],
                'length_of_service':[self.length_of_service],
                'kpi_80':[self.kpi_80],
                'award_won':[self.award_won],
                'avg_training_score':[self.avg_training_score],
                'sum_metric':[self.sum_metric],
                'total_score':[self.total_score]
            }


            df=pd.DataFrame(custom_Data_input_dict)
            logging.info("Data frame  done")

            return df
        

        except Exception as e:
            logging.info("Error getting at get data as dataframe")
            raise CustomException(e,sys)


                 


