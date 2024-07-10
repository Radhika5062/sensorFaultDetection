import sys

import numpy as np
import pandas as pd
from imblearn.combine import SMOTETomek
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline

from sensor.constant.training_pipeline import TARGET_COLUMN
from sensor.entity.artifact_entity import DataTransformationArtifact, DataValidationArtifact
from sensor.entity.config_entity import DataTransformationConfig
from sensor.exception import SensorException
from sensor.logger import logging
from sensor.ml.model.estimator import TargetValueMapping
from sensor.utils.main_utils import save_numpy_array_data, save_object


class DataTransformation:
    def __init__(self, data_validation_artifact:DataValidationArtifact,
                 data_transformation_config:DataTransformationConfig):
        '''
            :param data_validation_artifact: output reference of data ingestion artifact stage
            :param data_transformation_config: configuration for data transformation
        '''
        try:
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
        except Exception as e:
            raise SensorException(e, sys)
        
    @staticmethod    # this is made static so that we are able to access it anywhere and we will not need to create a class
    # variable to access it
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise SensorException(e, sys)
    
    @classmethod
    # In this method we are applying the best transformations that we have found from the EDA process done in notebook.
    # Output of this will be a pkl file. We will be saving this file later on. 
    def get_data_transformer_object(cls) -> Pipeline:
        try:
            robust_scaler = RobustScaler()
            simple_imputer = SimpleImputer(strategy='constant', fill_value=0)
            preprocessor = Pipeline(
                steps = [
                    ("Imputer", simple_imputer), # replaces missing values with zero
                    ("RobustScaler", robust_scaler) # keep every feature in same range and handle outlier
                ]
            )
            return preprocessor
        except Exception as e:
            raise SensorException(e, sys)
    

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
             # We will first read the test and train datasets
             train_df = DataTransformation.read_data(self.data_validation_artifact.valid_train_file_path)
             test_df = DataTransformation.read_data(self.data_validation_artifact.valid_test_file_path)

             # Now we are invoking the get_data_transformer_object method
             preprocessor = self.get_data_transformer_object()

             # Splitting the dependent and independent features from the training dataset
             input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis = 1)
             target_feature_train_df = train_df[TARGET_COLUMN]
             # Converting target column which is currently categorical into a numerical column by using mapping
             # defined in the estimator.py file
             target_feature_train_df = target_feature_train_df.replace(TargetValueMapping().to_dict())

             # Repeating all the above steps for test dataset
             input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis =1)
             target_feature_test_df = test_df[TARGET_COLUMN]
             target_feature_test_df = target_feature_test_df.replace(TargetValueMapping().to_dict())

             # Now we will do fit on the train dataset and transform on both the train dataset as well as the test dataset
             preprocessor_object = preprocessor.fit(input_feature_train_df)
             transformed_input_train_feature = preprocessor_object.transform(input_feature_train_df)
             transformed_input_test_feature = preprocessor_object.transform(input_feature_test_df)

             # Now we will handle the imbalance dataset using smotetomek
             smt  = SMOTETomek(sampling_strategy='minority')

             input_feature_train_final, target_feature_train_final = smt.fit_resample(
                                                                        transformed_input_train_feature,
                                                                        target_feature_train_df
             )
             input_feature_test_final, target_feature_test_final = smt.fit_resample(
                                                                        transformed_input_test_feature,
                                                                        target_feature_test_df
             )

             # Converting the data into array
             train_arr = np.c_[input_feature_train_final, np.array(target_feature_train_final)]
             test_arr = np.c_[input_feature_test_final, np.array(target_feature_test_final)]

             # Saving numpy arrays
             save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array = train_arr)
             save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)
             save_object( self.data_transformation_config.transformed_object_file_path, preprocessor_object,)

             # Preparing artifact
             data_transformation_artifact = DataTransformationArtifact(
                 transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                 transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                 transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
             )
             logging.info(f"Data transformation artifact: {data_transformation_artifact}")
             return data_transformation_artifact
        except Exception as e:
            raise SensorException(e, sys)




