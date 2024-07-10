from sensor.exception import SensorException
from sensor.logger import logging
from sensor.entity.artifact_entity import PredictionArtifact, DataTransformationArtifact, ModelTrainerArtifact
from sensor.entity.config_entity import PredictionConfig
from sensor.constant import training_pipeline
from sensor.ml.model.estimator import TargetValueMapping
from sensor.utils.main_utils import read_yaml, load_object
import pandas as pd
import numpy as np
import sys 
import os

from sensor.constant.training_pipeline import SCHEMA_FILE_PATH

class Prediction:
    def __init__(self, prediction_config:PredictionConfig
                #  data_transformation_artifact:DataTransformationArtifact,
                #  model_trainer_artifact:ModelTrainerArtifact
                ):
        try:
            logging.info("Entered the init method of Prediction class in the components")
            self.prediction_config = prediction_config
            self._schema_config = read_yaml(training_pipeline.SCHEMA_FILE_PATH)
            # self.data_transformation_artifact = data_transformation_artifact
            # self.model_trainer_artifact = model_trainer_artifact
        except Exception as e:
            raise SensorException(e, sys)
    
    def read_prediction_file_and_convert_to_dataframe(self):
        try:
            # get the prediction file path
            file_path = self.prediction_config.get_prediction_file_path
            logging.info("Entered the read_prediction_file_and_convert_to_dataframe method in component")
            logging.info(f"File name is {file_path}")
            # convert the
            df = pd.read_csv(file_path, header = 0)
            df.replace({"na":np.nan}, inplace = True)
            # df = pd.read_csv(file_path, header = 0, names=self._schema_config["columns_name"])
            logging.info(f"dataframe is {df}")
            logging.info(f"shape of dataframe is {df.shape}")
            return df
        except Exception as e:
            raise SensorException(e, sys)
    
    def drop_columns_from_prediction_dataframe(self, df):
        try:
            logging.info("Entered the drop_columns_from_prediction_dataframe in components")
            logging.info(f"Shape before: {df.shape}")
            df = df.drop(self._schema_config["drop_columns"], axis = 1)
            logging.info(f"Shape after: {df.shape}")
            return df
        except Exception as e:
            raise SensorException(e, sys)

    



    
    def initiate_prediction(self)->PredictionArtifact:
        try:
            logging.info("Entered the initiate_prediction in component")
            dataframe = self.read_prediction_file_and_convert_to_dataframe()
            logging.info("Starting dropping columns")
            dataframe = self.drop_columns_from_prediction_dataframe(dataframe)
            logging.info("Dropping columns completed in prediction")
            # dataframe= dataframe.replace({"na":np.nan}, inplace = True)
            # # dataframe = dataframe.reshape(-1, 1)
            # latest_preprocessed_object_path = self.get_best_preprocessed_object_path()
            # preprocessor =  load_object(file_path=latest_preprocessed_object_path)
            # fitted_dataframe = preprocessor.transform(dataframe)
            # model = load_object(file_path=self.prediction_config.trained_model_file_path)
            # predict_result = model.predict(fitted_dataframe)
            # dataframe['predicted_coulumn'] = predict_result
            
            # predicted_column = dataframe['predicted_coulumn'].replace(TargetValueMapping().reverse_mapping, inplace = True)

            prediction_artifact = PredictionArtifact(prediction_data=dataframe)
            logging.info(f"Prediction artifact before returning it is {prediction_artifact}")
            return prediction_artifact
        except Exception as e:
            raise SensorException(e, sys)





