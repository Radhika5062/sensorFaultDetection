from dotenv import load_dotenv
import pymongo.mongo_client
from sensor.constant.database import DATABASE_NAME
import os 
from sensor.constant.env_variable import MONGODB_URL_KEY
from sensor.logger import logging
import pymongo
import certifi
ca = certifi.where()

# This is used to call the .env file. 
load_dotenv()

class MongoDBClient:
    client = None # Creating a class variable

    def __init__(self, database_name = DATABASE_NAME) -> None: 
        try:
            if MongoDBClient.client is None:
                mongo_db_url =  os.getenv(MONGODB_URL_KEY)
                logging.info(f"Retrived mongo db url")

                if "localhost" in mongo_db_url:
                    MongoDBClient.client = pymongo.MongoClient(mongo_db_url)
                else:
                    MongoDBClient.client = pymongo.MongoClient(mongo_db_url, tlsCAFile = ca)
            
            self.client = MongoDBClient.client
            self.database = self.client[database_name]
            self.database_name = database_name
        except Exception as e:
            logging.info(f'Error initializing MongoDB client: {e}')
            raise e