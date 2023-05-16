import os,sys
from employee.exception import CustomException
from employee.logger import logging
from employee.constant import *
from employee.config.configuration import PREPROCESSING_OBJ_PATH
from dataclasses import dataclass


from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,PowerTransformer,OrdinalEncoder,OneHotEncoder
from sklearn.pipeline import Pipeline
from employee.utils.utils import save_object
from imblearn.combine import SMOTETomek
from sklearn.impute import IterativeImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge

@dataclass
class DataTransformationConfig():
    preprocessor_obj_file_path=PREPROCESSING_OBJ_PATH


class DataTransformation():
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()


    def get_data_transformation_object(self):
        try:
            logging.info("Loading data transformation")
            
# ordinal
            education=["Below Secondary", "Bachelor's", "Master's & above"]
            ordinal_encod=['education']
# numarical
            numerical_columns = ['no_of_trainings','age','previous_year_rating','length_of_service',
                                 'kpi_80','award_won','avg_training_score ']
# categorical features
            categorical_column =['gender']



                

            
            
            numerical_pipeline=Pipeline(steps=[
                ('impute',IterativeImputer(estimator=BayesianRidge(), initial_strategy='mean', n_nearest_features=None, 
                                           imputation_order='ascending')),
                ('scaler',StandardScaler()),
                ('transformer', PowerTransformer(method='yeo-johnson', standardize=False))
            ])

            oridinal_pipeline=Pipeline(steps=[
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('ordinal',OrdinalEncoder(categories=[education])),
                ('scaler',StandardScaler(with_mean=False))  

            ])

            categorical_pipeline=Pipeline(steps=[
                ('impute',SimpleImputer(strategy='most_frequent')),
                ('onehot',OneHotEncoder(handle_unknown='ignore')),
                ('scaler',StandardScaler(with_mean=False))
                ])

            preprocessor =ColumnTransformer([
                ('numerical_pipeline',numerical_pipeline,numerical_columns),
                ('ordinal_pipeline',oridinal_pipeline,ordinal_encod),
                ('category_pipeline',categorical_pipeline,categorical_column)
            ])

            return preprocessor

            logging.info('pipeline completed')


        except Exception as e:
            logging.info("Error getting data transformation object")
            raise CustomException(e,sys)
        


    def _remove_outliers_IQR(self, col, df):
        try:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            iqr = Q3 - Q1
            upper_limit = Q3 + 1.5 * iqr
            lower_limit = Q1 - 1.5 * iqr
            df.loc[(df[col]>upper_limit), col]= upper_limit
            df.loc[(df[col]<lower_limit), col]= lower_limit 
            return df
        
        except Exception as e:
            logging.info(" outlier code")
            raise CustomException(e, sys) from e 
        
   
        
    
    def initaite_data_transformation(self,train_path,test_path):
        try:
            # Reading train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data completed')
            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head  : \n{test_df.head().to_string()}')

            
            train_precent = ((train_df.isnull().sum() / train_df.shape[0])*100).round(2)
            logging.info(f" total nulll percentage : {train_precent}")


# missing value in education and previous year training

            train_df['education'] = train_df['education'].fillna(train_df['education'].mode()[0])
            train_df['previous_year_rating'] = train_df['previous_year_rating'].fillna(train_df['previous_year_rating'].mode()[0])

            logging.info("fill value in train.csv")

            logging.info(f"unique value in eduction {train_df['education'].unique() }")
            logging.info(f"unique value in previous_year_rating {train_df['previous_year_rating'].unique() }")


            test_df['education'] = test_df['education'].fillna(test_df['education'].mode()[0])
            test_df['previous_year_rating'] = test_df['previous_year_rating'].fillna(test_df['previous_year_rating'].mode()[0])

            logging.info("fill value in test.csv")

            logging.info(f"unique value in eduction {test_df['education'].unique() }")
            logging.info(f"unique value in previous_year_rating {test_df['previous_year_rating'].unique() }")

# numerical column
            numerical_columns = ['no_of_trainings','age','previous_year_rating','length_of_service',
                                 'kpi_80','award_won','avg_training_score ']
           

            # Assuming 'df' is your DataFrame
            num_col = [feature for feature in train_df.columns if train_df[feature].dtype != '0']
            


            logging.info(f"numerical_columns: {num_col}")

            for col in numerical_columns:
                self._remove_outliers_IQR(col=col, df= train_df)
            
            logging.info(f"Outlier capped in train df")
            
            for col in numerical_columns:
                self._remove_outliers_IQR(col=col, df= test_df)
                
            logging.info(f"Outlier capped in test df") 

            preprocessing_obj = self.get_data_transformation_object()

            logging.info(f"Train Dataframe Head:\n{train_df.head().to_string()}")
            logging.info(f"Test Dataframe Head:\n{test_df.head().to_string()}")

            target_column_name = 'concrete_compressive_strength'

            X_train = train_df.drop(columns=target_column_name,axis=1)
            y_train = train_df[target_column_name]

            X_test = test_df.drop(columns=target_column_name,axis=1)
            y_test = test_df[target_column_name]


            logging.info(f"shape of {X_train.shape} and {y_train.shape}")
            logging.info(f"shape of {X_test.shape} and {y_test.shape}")

            # Transforming using preprocessor obj
            
            X_train=preprocessing_obj.fit_transform(X_train)            
            X_test=preprocessing_obj.transform(X_test)

            logging.info("Applying preprocessing object on training and testing datasets.")
            logging.info(f"shape of {X_train.shape} and {y_train.shape}")
            logging.info(f"shape of {X_test.shape} and {y_test.shape}")

            smt = SMOTETomek(random_state=42,sampling_strategy='minority')
            
            input_feature_train_arr, target_feature_train_df = smt.fit_resample(X_train, y_train)
            
            input_feature_test_arr, target_feature_test_df = smt.fit_resample(X_test , y_test)
            


            logging.info("transformation completed")



            train_arr =np.c_[X_train,np.array(y_train)]
            test_arr =np.c_[X_test,np.array(y_test)]

            logging.info("train arr , test arr")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj)
            
            logging.info("Preprocessor file saved")
            
            return(train_arr,
                   test_arr,
                   self.data_transformation_config.preprocessor_obj_file_path)

        except Exception as e:
            raise CustomException(e, sys) from e 

