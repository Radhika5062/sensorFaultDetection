from sensor.exception import SensorException
from sensor.logger import logging
import os
import sys 
import numpy as np
from utils import dump_csv_file_to_mongodb_collection
from sensor.pipeline.training_pipeline import TrainPipleine


# Fast api related stuff
from sensor.constant.application import APP_HOST, APP_PORT
from fastapi import FastAPI, File, UploadFile,Response
from starlette.responses import RedirectResponse
from uvicorn import run as app_run
from fastapi.responses import Response
from sensor.ml.model.estimator import ModelResolver, TargetValueMapping
from sensor.utils.main_utils import load_object
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from sensor.pipeline.prediction_pipeline import PredictionPipeline
from sensor.constant.training_pipeline import SAVED_MODEL_DIR

# def test_exception():
#     try:
#         logging.info("Trsting logging")
#         a = 1/0
#     except Exception as e:
#         raise SensorException(e,sys)

# if __name__ == '__main__':
#     try:
#         test_exception()
#     except Exception as e:
#         print(e)


app = FastAPI()

origins = ["*"]
# Cross origin resource sharing (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins = origins,
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"],
)

# in this one, we are going to write code about connecting with fast api. 
@app.get("/", tags=["authentication"])
async def index():
    # If we go to this "/" url directly then we will get data printed in regular format. But as we need the data to be in
    # swagger file format therefore we are adding /docs so that the layout of the printed page is like swagger.
    return RedirectResponse(url="/docs")

# In this, one we will write code about training
@app.get("/train")
async def train():
    try:
        logging.info("Hitting train")
        training_pipeline = TrainPipleine()
        if training_pipeline.is_pipeline_running:
            return Response("Training pipeline is already running")
        training_pipeline.run_pipeline()
        return Response("Training pipeline completed")
    except Exception as e:
        return Response(f"Error occurred {e}")

# In this, we will write code about prediction
# @app.get("/test")
# async def test():
#     try:
#         return Response(f"Print me")
#     except Exception as e:
#         return Response(f"Errorrr")

    
@app.get("/predict")
async def predict():
    try:
        logging.info("Entered the prediction function in main.py")
        prediction_pipeline = PredictionPipeline()
        logging.info("Running prediction pipeline in main.py")
        prediction_artifact = prediction_pipeline.run_pipeline()
        df = prediction_artifact.prediction_data
        logging.info(f"My dataframe is {df}")

        Model_resolver = ModelResolver(model_dir=SAVED_MODEL_DIR)
        if not Model_resolver.is_model_exists():
            return Response("Model is not available")
        
        best_model_path = Model_resolver.get_best_model_path()
        model= load_object(file_path=best_model_path)
        y_pred=model.predict(df)
        df = pd.DataFrame({'predicted_column':y_pred})
        mapping_instance = TargetValueMapping()
        reverse_mapping_dict = mapping_instance.reverse_mapping()
        df['predicted_column'].replace(reverse_mapping_dict, inplace = True)
        return Response(f"Output is: {df['predicted_column']}")
    except  Exception as e:
        raise  SensorException(e,sys)

def main():
    try:
        training_pipeline = TrainPipleine
        training_pipeline.run_pipeline()
    except Exception as e:
        raise SensorException(e, sys)

if __name__=='__main__':
    # file_path = "C:/Users/RadhikaMaheshwari/Desktop/Test/DeepLearning/ETE2/aps_failure_training_set1.csv"
    # database_name = "ineuron"
    # collection_name = 'sensor'
    # dump_csv_file_to_mongodb_collection(file_path, database_name, collection_name)
    # training_pipeline = TrainPipleine()
    # training_pipeline.run_pipeline()
    # added comment port
    app_run(app,host=APP_HOST,port=APP_PORT)