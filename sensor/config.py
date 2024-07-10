from dataclasses import dataclass
import os
import pymongo 

@dataclass

class EnvironmentVariable:
    mongo_db_url:str = os.getenv("MONGO_DB_URL") # This is a constant which we have defined in the .env file. 

# Now we create an object of this class
env_var = EnvironmentVariable()

# Now we need to create the mongo client
# env_var.mongo_db_url is the way we can access the class variable. object.variable name
mongo_client = pymongo.MongoClient(env_var.mongo_db_url)
