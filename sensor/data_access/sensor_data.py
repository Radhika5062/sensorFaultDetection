from sensor.constant.database import DATABASE_NAME
from sensor.configuration.mongo_db_connection import MongoDBClient
from sensor.exception import SensorException
from typing import Optional
import sys 
import pandas as pd
import json
import numpy as np
from sensor.logger import logging

class SensorData:
    def __init__(self):
        try:
            self.mongo_client = MongoDBClient(database_name = DATABASE_NAME)
        except Exception as e:
            raise SensorException(e, sys)
        
    def save_csv_file(self, file_path, collection_name:str, database_name: Optional[str] = None):
        try:
            data_frame = pd.read_csv(file_path)
            data_frame.reset_index(drop = True, inplace = True)
            records = list(json.loads(data_frame.T.to_json()).values())
            if database_name is None:
                collection = self.mongo_client.database[collection_name]
            else:
                collection = self.mongo_client[database_name][collection_name]
            collection.insert_many(records)
        except Exception as e:
            raise SensorException(e, sys)
        
    def export_collection_as_dataframe(self, collection_name:str, database_name: Optional[str] = None) -> pd.DataFrame:
        try:
            logging.info(f"database name in export_collection_as_dataframe method in sensor_data.py file is {database_name}")
            if database_name is None:
                collection = self.mongo_client.database[collection_name]
            else:
                collection = self.mongo_client[database_name][collection_name]
            df = pd.DataFrame(list(collection.find()))
            logging.info(f"dataframe created in export_collection_as_dataframe method in sensor_data.py file is {df.shape} shape originally")

            if "_id" in df.columns.to_list():
                df = df.drop(columns=["_id"], axis = 1)
            
            df.replace({"na":np.nan}, inplace = True)
            logging.info(f"dataframe created in export_collection_as_dataframe method in sensor_data.py file is {df.shape} shape at the time of exit")
            return df
        except Exception as e:
            raise SensorException(e,sys)