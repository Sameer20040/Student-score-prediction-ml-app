import sys 
from dataclasses import dataclass
import os 

import numpy as np 
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logger

from src.utils import save_object

# defining preprocessing object will be saved and where it will be saved
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str=os.path.join('artifacts','preprocessor.pkl')
# handles all data transformation tasks and preprocessing logic and steps
class DataTransformation:
    def __init__(self): # makes file path accessible inside the class
        self.data_transformation_config=DataTransformationConfig()
    
    # this method creates and returns the preprocessing pipline and object
    def get_data_transformer_object(self):
        # This function is responsible for data transformation
        try:
            numerical_columns=['writing_score','reading_score']
            categorical_columns=[ 'gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']

            num_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')), # replace missing values with median
                    ('scaler',StandardScaler())# standardize the numerical columns mean=0 variance=1
                ]
            )

            cat_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder',OneHotEncoder(handle_unknown='ignore')), # convert categorical columns to numerical using one hot encoding and avoids errors for unseen categories
                    ('scaler',StandardScaler(with_mean=False, with_std=True)) # with_mean=False to avoid centering sparse matrix
                ]
            )
            logger.info('Numerical and categorical pipelines created')
            logger.info(f'Numerical columns: {numerical_columns}')
            logger.info(f'Categorical columns: {categorical_columns}')

            # Combining both numerical and categorical pipelines using into one preprocessinf step
            preprocessor=ColumnTransformer(
                transformers=[
                    ('num_pipeline',num_pipeline,numerical_columns),
                    ('cat_pipeline',cat_pipeline,categorical_columns)
                ]
            )

            return preprocessor


        except Exception as e:
            logger.info("Exception occurred in data transformation")
            raise CustomException(e,sys)


    # main function to initiate data transformation
    def initiate_data_transformation(self,train_path,test_path):
        try:
            # Reading train and test data
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logger.info("Read train and test data completed")

            logger.info("Obtaining preprocessor object")
 
            # builds the preprocessing pipeline
            preprocessing_obj=self.get_data_transformer_object()
            target_column_name='math_score'
            numerical_columns=['writing_score','reading_score']

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logger.info("Applying preprocessing object on training and testing datasets")
    
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)
            logger.info("Transformation complete")
           

           # combined transforemd features and target column into oen numpy array and this required for model training pipeline
            train_arr=np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_df)]
            
            logger.info("Saved preprocessing object")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )


            # preprocess training and testing array along with the path of saved preprocessor
            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            logger.info("Exception occurred in initiate_data_transformation")
            raise CustomException(e,sys)



# reads raw data 
# Hndles missing values
# encodes categorical features
# scales numerical features
# saves the preprocessor object
# returns transformed training and testing arrays