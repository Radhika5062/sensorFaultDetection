from sensor.entity.config_entity import TrainingPipelineConfig, DataIngestionConfig, \
DataValidationConfig, DataTransformationConfig, ModelTrainerConfig, ModelEvaluationConfig, ModelPusherConfig
from sensor.exception import SensorException
from sensor.logger import logging
from sensor.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact, DataTransformationArtifact, \
ModelTrainerArtifact, ModelEvaluationArtifact
from sensor.components.data_ingestion import DataIngestion
from sensor.components.data_validation import DataValidation
from sensor.components.data_transformation import DataTransformation
from sensor.components.model_trainer import ModelTrainer
from sensor.components.model_evaluation import ModelEvaluation
from sensor.components.model_pusher import ModelPusher
from sensor.constant.training_pipeline import SAVED_MODEL_DIR
import os, sys 

# Deployment related code
from sensor.cloud_storage.s3_syncer import S3Sync
from sensor.constant.s3_bucket import TRAINING_BUCKET_NAME
#

class TrainPipleine:
    # Set up a flag to understand if training pipeline is running or not. Currently, it will be set as False as
    # we are defininf it in class and even if the code comes to this class does not mean that the training has started
    # We will set it to True or false in run section of this class as that will essentially define whether the pipeline is 
    # running or not. So essentially, the main reason why we are defining this here is so that it can be created as a 
    # class variable and we can then use it outside of the class too. 
    is_pipeline_running = False
    #Deployment related code
    s3_sync = S3Sync()
    #
    def __init__(self):
        self.training_pipeline_config = TrainingPipelineConfig()
    
    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            self.data_ingestion_config = DataIngestionConfig(training_pipeline_config=self.training_pipeline_config)
            logging.info("Starting data ingestion")

            data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()

            logging.info(f"Data ingestion completed and artifact {data_ingestion_artifact}")
            return data_ingestion_artifact
        except Exception as e:
           raise SensorException(e, sys)
    
    def start_data_validation(self, data_ingestion_artifact:DataIngestionArtifact) -> DataValidationArtifact:
        try:
            data_validation_cofig = DataValidationConfig(training_pipeline_config=self.training_pipeline_config)

            data_validation = DataValidation(data_ingestion_artifact=data_ingestion_artifact, 
                                             data_validation_config= data_validation_cofig)
            data_validation_artifact = data_validation.initiate_data_validation()
            return data_validation_artifact
        except Exception as e:
            raise SensorException(e, sys)
    
    def start_data_transformation(self, data_validation_artifact:DataValidationArtifact):
        try:
            data_transformation_config = DataTransformationConfig(training_pipeline_config=self.training_pipeline_config)
            data_transformation = DataTransformation(data_validation_artifact=data_validation_artifact,
                                                     data_transformation_config=data_transformation_config)
            data_transformation_artifact = data_transformation.initiate_data_transformation()
            return data_transformation_artifact
        except Exception as e:
            raise SensorException(e, sys)
    
    def start_model_trainer(self,data_transformation_artifact:DataTransformationArtifact):
        try:
            model_trainer_config = ModelTrainerConfig(training_pipeline_config=self.training_pipeline_config)
            model_trainer = ModelTrainer(model_trainer_config, data_transformation_artifact)
            model_trainer_artifact = model_trainer.initiate_model_trainer()
            return model_trainer_artifact
        except Exception as e:
            raise SensorException(e, sys)
        
    def start_model_evaluation(self, 
                               data_validation_artifact: DataValidationArtifact,
                               model_trainer_artifact:ModelTrainerArtifact):
        try:
            model_eval_config = ModelEvaluationConfig(self.training_pipeline_config)
            model_eval = ModelEvaluation(
                            model_eval_config,
                            data_validation_artifact,
                            model_trainer_artifact
            )
            model_eval_artifact = model_eval.initiate_model_evaluation()
            return model_eval_artifact
        except Exception as e:
            raise SensorException(e, sys)
    
    def start_model_pusher(self,
                           model_eval_artifact:ModelEvaluationArtifact):
        try:
            model_pusher_config = ModelPusherConfig(training_pipeline_config=self.training_pipeline_config)
            model_pusher = ModelPusher(model_pusher_config, model_eval_artifact)
            model_pusher_artifact = model_pusher.initiate_model_pusher()
            return model_pusher_artifact
        except Exception as e:
            raise SensorException(e, sys)
    
    # Deployment related code
    def sync_artifact_dir_to_s3(self):
        try:
            aws_bucket_url = f"s3://{TRAINING_BUCKET_NAME}/artifact/{self.training_pipeline_config.timestamp}"
            self.s3_sync.sync_folder_to_s3(folder = self.training_pipeline_config.artifact_dir, aws_bucket_url=aws_bucket_url)
        except Exception as e:
            raise SensorException(e, sys)
    
    def sync_saved_model_dir_to_s3(self):
        try:
            aws_bucket_url = f"s3://{TRAINING_BUCKET_NAME}/{SAVED_MODEL_DIR}"
            self.s3_sync.sync_folder_to_s3(folder = SAVED_MODEL_DIR, aws_bucket_url=aws_bucket_url)
        except Exception as e:
            raise SensorException(e, sys)
    #


    def run_pipeline(self):
        try:
            # as we are entering the run part which means we are initiating the training pipeline so we will
            # assign is_pipeline_running as True. 
            TrainPipleine.is_pipeline_running = True
            data_ingestion_artifact:DataIngestionArtifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)
            data_transformation_artifact = self.start_data_transformation(data_validation_artifact=data_validation_artifact)
            model_trainer_artifact = self.start_model_trainer(data_transformation_artifact=data_transformation_artifact)
            model_evaluation_artifact = self.start_model_evaluation(data_validation_artifact, model_trainer_artifact)

            if not model_evaluation_artifact.is_model_accpeted:
                raise Exception("Trained model us not better than best model")
            
            model_pusher_artifact = self.start_model_pusher(model_evaluation_artifact)
            # now that we are exiting the run part so we will again set the is_pipeline running to False so that we can start it 
            # later

            # Deployment code
            self.sync_artifact_dir_to_s3()
            self.sync_saved_model_dir_to_s3()

            #
            TrainPipleine.is_pipeline_running = False
        except Exception as e:
            # If we encounter any error then also the training pipeline variable should be set to False so we will do it here.
            # Deployment code
            self.sync_artifact_dir_to_s3()
            #  
            TrainPipleine.is_pipeline_running = False
            raise SensorException(e, sys)