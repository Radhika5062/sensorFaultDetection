from sensor.entity.config_entity import ModelPusherConfig
from sensor.entity.artifact_entity import ModelEvaluationArtifact, ModelPusherArtifact
from sensor.exception import SensorException

import os
import shutil
import sys



class ModelPusher:
    def __init__(self,
                 model_pusher_config:ModelPusherConfig,
                 model_eval_artifact:ModelEvaluationArtifact):
        try:
            self.model_pusher_config = model_pusher_config
            self.model_eval_artifact = model_eval_artifact
        except Exception as e:
            raise SensorException(e, sys)
    
    def initiate_model_pusher(self)->ModelPusherArtifact:
        try:
            trained_model_path = self.model_eval_artifact.trained_model_path

            # Creating model pusher directory to save model
            model_file_path = self.model_pusher_config.model_file_path
            os.makedirs(os.path.dirname(model_file_path), exist_ok = True)
            shutil.copy(src = trained_model_path, dst=model_file_path)

            # Saved model dir
            saved_model_path = self.model_pusher_config.saved_model_path
            os.makedirs(os.path.dirname(saved_model_path), exist_ok=True)
            shutil.copy(src = trained_model_path, dst = saved_model_path)

            # Prepare artifact
            model_pusher_artifact = ModelPusherArtifact(
                                                model_file_path=model_file_path,
                                                saved_model_path=saved_model_path
            )

            return model_pusher_artifact
        except Exception as e:
            raise SensorException(e, sys)