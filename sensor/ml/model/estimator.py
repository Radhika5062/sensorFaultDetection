import os
from sensor.exception import SensorException
import sys
from sensor.constant.training_pipeline import SAVED_MODEL_DIR, MODEL_FILE_NAME

class TargetValueMapping:
    def __init__(self):
        self.neg: int = 0
        self.pos: int = 1
    
    def to_dict(self):
        return self.__dict__
    
    def reverse_mapping(self):
        mapping_response = self.to_dict()
        return dict(zip(mapping_response.values(), mapping_response.keys()))

class SensorModel:
    def __init__(self, preprocessor, model):
        try:
            self.preprocessor = preprocessor
            self.model = model
        except Exception as e:
            raise SensorException(e, sys)
    
    def predict(self,x):
        try:
            x_transform = self.preprocessor.transform(x)
            y_hat = self.model.predict(x_transform)
            return y_hat
        except Exception as e:
            raise SensorException(e, sys)

class ModelResolver:
    # First we are fetching the saved model
    def __init__(self, model_dir = SAVED_MODEL_DIR):
        try:
            self.model_dir = model_dir
        except Exception as e:
            raise SensorException(e, sys)
    
    def get_best_model_path(self)->str:
        try:
            # There will be multiple models accumulated over time. therefore the below code will first list them and then
            # choose the file which has the highest timestamp so that we know this is the latest model.
            timestamps = list(map(int,os.listdir(self.model_dir)))
            latest_timestamp = max(timestamps)
            latest_model_path = os.path.join(self.model_dir,f"{latest_timestamp}", MODEL_FILE_NAME)
            return latest_model_path
        except Exception as e:
            raise SensorException(e, sys)
        
    # now we need to check whether model exists or not. first this function will be called and then get_best_model_path
    # will be called. How this function works is:
    # 1. First we check whether we have a model directory or not.
    # 2. If saved model directory doesn't exist then we return false. 
    # 3. If it exists then we check whether it is empty.
    # 4. If it is empty then we return false. 
    # 5. If it is not empty then we call the get_best_model_path function to get the latest model path from all the 
    # paths that are present.
    def is_model_exists(self)->bool:
        try:
            if not os.path.exists(self.model_dir):
                return False
            timestamps = os.listdir(self.model_dir)
            if len(timestamps) == 0:
                return False
            
            latest_model_path = self.get_best_model_path()

            if not os.path.exists(latest_model_path):
                return False
            
            return True
        except Exception as e:
            raise SensorException(e, sys)