import os,sys
from employee.exception import CustomException
from employee.logger import logging
from employee.constant import *
from employee.config.configuration import PREPROCESSING_OBJ_PATH,TRANSFORMED_TRAIN_FILE_PATH,TRANSFORMED_TEST_FILE_PATH,FEATURE_ENG_OBJ_PATH
from dataclasses import dataclass

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,PowerTransformer,OrdinalEncoder,OneHotEncoder
from sklearn.pipeline import Pipeline
from employee.utils.utils import save_object
from imblearn.combine import SMOTETomek



class Feature_Engineering(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        
        """
        This class applies necessary Feature Engneering 
        """
        logging.info(f"\n{'*'*20} Feature Engneering Started {'*'*20}\n\n")


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
        


    def transform_data(self,df):
        try:
            df['sum_metric'] = df['award_won'] + df['kpi_80'] + df['previous_year_rating']
            df['total_score'] = df['avg_training_score'] * df['no_of_trainings']

            logging.info("new columns sum_metric , total_score")


            # Assuming 'df' is your DataFrame
            num_col = [feature for feature in df.columns if df[feature].dtype != '0']
            
            logging.info(f"numerical_columns: {num_col}")


            cat_col = [feature for feature in df.columns if df[feature].dtype == 'O']
            logging.info(f"categorical_columns: {cat_col}")

            


            df.drop(columns=['region','recruitment_channel'], inplace=True, axis=1)

            logging.info(f"columns in dataframe are: {df.columns}")


            numerical_columnss = [ 'no_of_trainings', 'age', 'previous_year_rating',
                                  'length_of_service', 'kpi_80', 'award_won', 'avg_training_score','sum_metric', 'total_score']


# outlier

            for col in numerical_columnss:
                self._remove_outliers_IQR(col=col, df= df)
            
            logging.info(f"Outlier capped in train df")



# missing value in education and previous year training

            df['education'] = df['education'].fillna(df['education'].mode()[0])
            df['previous_year_rating'] = df['previous_year_rating'].fillna(df['previous_year_rating'].mode()[0])

            logging.info("fill value in df.csv")

            logging.info(f"unique value in eduction {df['education'].unique() }")
            logging.info(f"unique value in previous_year_rating {df['previous_year_rating'].unique() }")

            logging.info(f"Train Dataframe Head:\n{df.head().to_string()}")
            #logging.info(f"Test Dataframe Head:\n{test_df.head().to_string()}")
     
            return df

         
        except Exception as e:
            logging.info(" transform and add new columns error occurred")
            raise CustomException(e, sys) from e





    def fit(self,X,y=None):
        return self
    
    
    def transform(self,X:pd.DataFrame,y=None):
        try:    
            transformed_df=self.transform_data(X)
                
            return transformed_df
        except Exception as e:
            raise CustomException(e,sys) from e











@dataclass
class DataTransformationConfig():
    preprocessor_obj_file_path=PREPROCESSING_OBJ_PATH
    transformed_train_path=TRANSFORMED_TRAIN_FILE_PATH
    transformed_test_path=TRANSFORMED_TEST_FILE_PATH
    feature_eng_obj_path=FEATURE_ENG_OBJ_PATH


class DataTransformation():
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()


    def get_data_transformation_object(self):
        try:
            logging.info("Loading data transformation")
            
# ordinal
            
            ordinal_columns=['education']
# numarical
            numerical_columns = ['no_of_trainings','age','previous_year_rating','length_of_service',
                                 'kpi_80','award_won','avg_training_score','sum_metric','total_score']
# categorical features
            categorical_columns =['gender','department']


            logging.info("ordinal_columns -  numerical_columns -  categorical_columns")
            
            
            numerical_pipeline=Pipeline(steps=[
                ('impute',SimpleImputer()),
                ('scaler',StandardScaler()),
                ('transformer', PowerTransformer(method='yeo-johnson', standardize=False))
            ])

            oridinal_pipeline=Pipeline(steps=[
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('ordinal',OrdinalEncoder()),
                ('scaler',StandardScaler(with_mean=False))  

            ])

            categorical_pipeline=Pipeline(steps=[
                ('impute',SimpleImputer(strategy='most_frequent')),
                ('onehot',OneHotEncoder(handle_unknown='ignore')),
                ('scaler',StandardScaler(with_mean=False))
                ])

            preprocessor =ColumnTransformer([
                ('numerical_pipeline',numerical_pipeline,numerical_columns),
                ('ordinal_pipeline',oridinal_pipeline,ordinal_columns),
                ('category_pipeline',categorical_pipeline,categorical_columns)
            ])

            return preprocessor

            logging.info('pipeline completed')


        except Exception as e:
            logging.info("Error getting data transformation object")
            raise CustomException(e,sys)
        


    def get_feature_engineering_object(self):
        try:
            
            feature_engineering = Pipeline(steps = [("fe",Feature_Engineering())])
            return feature_engineering
        except Exception as e:
            raise CustomException(e,sys) from e




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






# numerical column
            numerical_columns = ['no_of_trainings','age','previous_year_rating','length_of_service',
                                 'kpi_80','award_won','avg_training_score ']
           

           # Feature engimeering pipleine
                    # Feature Engineering 
            logging.info(f"Obtaining feature engineering object.")
            fe_obj = self.get_feature_engineering_object()

            logging.info(f"Applying feature engineering object on training dataframe and testing dataframe")
            logging.info(">>>" * 20 + " Training data " + "<<<" * 20)
            logging.info(f"Feature Enineering - Train Data ")
            train_df = fe_obj.fit_transform(train_df)
            logging.info(">>>" * 20 + " Test data " + "<<<" * 20)
            logging.info(f"Feature Enineering - Test Data ")
            test_df = fe_obj.transform(test_df)

            train_df.to_csv("train_data.csv")
            test_df.to_csv("test_data.csv")
            logging.info(f"Saving csv to train_data and test_data.csv")





            # Preprocessing pipeline 
            preprocessing_obj = self.get_data_transformation_object()

            



# target column -- is_promoted

            target_column_name = 'is_promoted'

            logging.info(f"shape of {train_df.shape} and {test_df.shape}")
            

            X_train = train_df.drop(columns=target_column_name,axis=1)
            y_train = train_df[target_column_name]

            X_test = test_df.drop(columns=target_column_name,axis=1)
            y_test = test_df[target_column_name]


            #logging.info(f"shape of {X_train.shape} and {y_train.shape}")
            #logging.info(f"shape of {X_test.shape} and {y_test.shape}")

            # Transforming using preprocessor obj
            logging.info(f"dataset column {X_train.columns}" )
            X_train=preprocessing_obj.fit_transform(X_train)            
            X_test=preprocessing_obj.transform(X_test)

            logging.info("Applying preprocessing object on training and testing datasets.")
            logging.info(f"shape of {X_train.shape} and {y_train.shape}")
            logging.info(f"shape of {X_test.shape} and {y_test.shape}")
            logging.info("****************************************************************************")


# sampling
            smt = SMOTETomek(random_state=42,sampling_strategy='minority')
            logging.info("****************************************************************************")
            input_feature_train_arr, target_feature_train_df = smt.fit_resample(X_train, y_train)
            
            input_feature_test_arr, target_feature_test_df = smt.fit_resample(X_test , y_test)
            


            logging.info("transformation completed")



            train_arr =np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr =np.c_[input_feature_test_arr,np.array(target_feature_test_df)]
            

            logging.info("train arr , test arr")


            df_train= pd.DataFrame(train_arr)
            df_test = pd.DataFrame(test_arr)

            logging.info("converting train_arr and test_arr to dataframe")
            #logging.info(f"Final Train Transformed Dataframe Head:\n{df_train.head().to_string()}")
            #logging.info(f"Final Test transformed Dataframe Head:\n{df_test.head().to_string()}")

            os.makedirs(os.path.dirname(self.data_transformation_config.transformed_train_path),exist_ok=True)
            df_train.to_csv(self.data_transformation_config.transformed_train_path,index=False,header=True)

            logging.info("transformed_train_path")
            logging.info(f"transformed dataset columns : {df_train.columns}")

            os.makedirs(os.path.dirname(self.data_transformation_config.transformed_test_path),exist_ok=True)
            df_test.to_csv(self.data_transformation_config.transformed_test_path,index=False,header=True)

            logging.info("transformed_test_path")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj)
            
            logging.info("Preprocessor file saved")

            save_object(
                file_path=self.data_transformation_config.feature_eng_obj_path,
                obj=fe_obj)
            logging.info("Feature eng file saved")


            
            return(train_arr,
                   test_arr,
                   self.data_transformation_config.preprocessor_obj_file_path)

        except Exception as e:
            raise CustomException(e, sys) from e 

