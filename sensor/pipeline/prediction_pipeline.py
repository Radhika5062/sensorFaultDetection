from sensor.entity.config_entity import PredictionConfig
from sensor.entity.artifact_entity import PredictionArtifact
from sensor.components.prediction import Prediction
from sensor.pipeline.training_pipeline import TrainingPipelineConfig
from sensor.logger import logging

class PredictionPipeline:
    def __init__(self):
        pass

    def start_prediction_pipeline(self):
        self.prediction_pipeline_config = PredictionConfig()
        prediction = Prediction(prediction_config=self.prediction_pipeline_config)

        prediction_artifact = prediction.initiate_prediction()
        logging.info(f"Prediction artifact in start_prediction_pipeline in prediction pipeline is {prediction_artifact}")

        return prediction_artifact

    
    def run_pipeline(self):
        prediction_artifact:PredictionArtifact = self.start_prediction_pipeline()
        return prediction_artifact
