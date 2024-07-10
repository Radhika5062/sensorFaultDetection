from datetime import datetime
import os
from sensor.exception import SensorException
from sensor.constant import training_pipeline

# In this all we are doing is creating the folder and file structure. 
# we need to read the path from constant>training_config init file and then just create the folder structure. 
# Structure is as follows:
# root dir : artifact
# artifact > timestamp
# timestamp > data transformation
# timestamp > data validation
# atifact > timestamp > ingested
# ingested > train,csv, test.csv, feature store folder
# Feature store folder > sensor.csv

class TrainingPipelineConfig:
    def __init__(self, timestamp=datetime.now()):
        timestamp = timestamp.strftime("%m_%d_%Y_%H_%M_%S")
        self.pipeline_name = training_pipeline.PIPELINE_NAME
        self.artifact_dir = os.path.join(training_pipeline.ARTIFACT_DIR, timestamp)
        self.timestamp:str = timestamp



class DataIngestionConfig:
    def __init__(self, training_pipeline_config:TrainingPipelineConfig):
        self.data_ingestion_dir:str =  os.path.join(
                                        training_pipeline_config.artifact_dir, 
                                        training_pipeline.DATA_INGESTION_DIR_NAME
                                        )
        self.feature_store_file_path:str = os.path.join(
                                        self.data_ingestion_dir,
                                        training_pipeline.DATA_INGESTION_FEATURE_STORE_DIR,
                                        training_pipeline.FILE_NAME
        )
        self.training_file_path:str = os.path.join(
                                        self.data_ingestion_dir, 
                                        training_pipeline.DATA_INGESTION_INGESTED_DIR,
                                        training_pipeline.TRAIN_FILE_NAME
        )
        self.testing_file_path:str = os.path.join(
                                        self.data_ingestion_dir,
                                        training_pipeline.DATA_INGESTION_INGESTED_DIR,
                                        training_pipeline.TEST_FILE_NAME
        )
        self.train_test_split_ratio:float = training_pipeline.DATA_INGESTION_TRAIN_TEST_SPLIT_RATION
        self.collection_name:str = training_pipeline.DATA_INGESTION_COLLECTION_NAME


class DataValidationConfig:
    def __init__(self, training_pipeline_config:TrainingPipelineConfig):
        self.data_validation_dir:str = os.path.join(
                                    training_pipeline_config.artifact_dir, 
                                    training_pipeline.DATA_VALIDATION_DIR_NAME
        )

        self.valid_data_dir:str = os.path.join(
                                    self.data_validation_dir,
                                    training_pipeline.DATA_VALIDATION_VALID_DIR
        )

        self.invalid_data_dir:str = os.path.join(
                                    self.data_validation_dir, 
                                    training_pipeline.DATA_VALIDATION_INVALID_DIR
        )
        
        self.valid_train_file_path:str = os.path.join(
            self.valid_data_dir, training_pipeline.TRAIN_FILE_NAME
        )

        self.valid_test_file_path:str = os.path.join(
            self.valid_data_dir,
            training_pipeline.TEST_FILE_NAME
        )

        self.drift_report_file_path:str = os.path.join(
            self.data_validation_dir,
            training_pipeline.DATA_VALIDATION_DRIFT_REPORT_DIR,
            training_pipeline.DATA_VALIDATION_DRIFT_REPORT_FILE_NAME
        )


class DataTransformationConfig:
    def __init__(self, training_pipeline_config:TrainingPipelineConfig):
        self.data_transformation_dir:str = os.path.join(
                                        training_pipeline_config.artifact_dir,
                                        training_pipeline.DATA_TRANSFORMATION_DIR_NAME
        )

        self.transformed_train_file_path:str = os.path.join(
                                        self.data_transformation_dir,
                                        training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
                                        training_pipeline.TRAIN_FILE_NAME.replace('csv','npy')
        )

        self.transformed_test_file_path:str = os.path.join(
                                        self.data_transformation_dir,
                                        training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
                                        training_pipeline.TEST_FILE_NAME.replace('csv','npy')
        )

        self.transformed_object_file_path:str = os.path.join(
                                        self.data_transformation_dir,
                                        training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,
                                        training_pipeline.PREPROCESSING_OBJECT_FILE_NAME
        )

class ModelTrainerConfig:
    def __init__(self, training_pipeline_config:TrainingPipelineConfig):
        self.model_trainer_dir:str = os.path.join(
                                        training_pipeline_config.artifact_dir,
                                        training_pipeline.MODEL_TRAINER_DIR_NAME
        )
        
        self.trained_model_file_path = os.path.join(
                                        self.model_trainer_dir,
                                        training_pipeline.MODEL_TRAINER_TRAINED_MODEL_DIR,
                                        training_pipeline.MODEL_FILE_NAME
        )

        self.expected_accuracy:float = training_pipeline.MODEL_TRAINER_EXPECTED_SCORE
        self.overfitting_underfitting_threshold = training_pipeline.MODEL_TRAINER_OVER_FITTING_UNDER_FITTING_THRESHOLD

class ModelEvaluationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.model_evaluation_dir:str = os.path.join(
                                            training_pipeline_config.artifact_dir,
                                            training_pipeline.MODEL_EVALUATION_DIR_NAME
        )
        self.report_file_path = os.path.join(
                                            self.model_evaluation_dir,
                                            training_pipeline.MODEL_EVALUATION_REPORT_NAME
        )
        self.change_threshold = training_pipeline.MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE


class ModelPusherConfig:
    def __init__(self, training_pipeline_config:TrainingPipelineConfig):
        # this is used for storing the model for short time for deployment
        self.model_pusher_dir:str = os.path.join(
                                                training_pipeline_config.artifact_dir,
                                                training_pipeline.MODEL_PUSHER_DIR_NAME
        )
        self.model_file_path = os.path.join(
                                                self.model_pusher_dir,
                                                training_pipeline.MODEL_FILE_NAME
        )

        # This is used for storing model long term.
        timestamp = round(datetime.now().timestamp())
        self.saved_model_path = os.path.join(
                            training_pipeline.SAVED_MODEL_DIR,
                            f"{timestamp}",
                            training_pipeline.MODEL_FILE_NAME
        )

class PredictionConfig:
    def __init__(self):
        self.artifact_dir = os.path.join(training_pipeline.ARTIFACT_DIR)

        self.get_prediction_file_path = os.path.join(training_pipeline.PREDICTION_DIR_NAME,
                                                     training_pipeline.PREDICTION_CSV_FILE_PATH
                                        
        )
        self.model_trainer_dir:str = os.path.join(
                                        training_pipeline.ARTIFACT_DIR,
                                        training_pipeline.MODEL_TRAINER_DIR_NAME
        )
        
        self.trained_model_file_path = os.path.join(
                                        training_pipeline.MODEL_TRAINER_DIR_NAME,
                                        training_pipeline.MODEL_TRAINER_TRAINED_MODEL_DIR,
                                        training_pipeline.MODEL_FILE_NAME
        )
        self.data_transformation_dir:str = os.path.join(
                                        training_pipeline.ARTIFACT_DIR,
                                        training_pipeline.DATA_TRANSFORMATION_DIR_NAME
        )
        self.transformed_object_file_path:str = os.path.join(
                                        self.data_transformation_dir,
                                        training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,
                                        training_pipeline.PREPROCESSING_OBJECT_FILE_NAME
        )
        self.partial_transformed_object_path:str = os.path.join(
                                        training_pipeline.DATA_TRANSFORMATION_DIR_NAME,
                                        training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,
                                        training_pipeline.PREPROCESSING_OBJECT_FILE_NAME
        )