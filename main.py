from sensor.exception import SensorException
from sensor.logger import logging
import os
import sys 

def test_exception():
    try:
        logging.info("Trsting logging")
        a = 1/0
    except Exception as e:
        raise SensorException(e,sys)

if __name__ == '__main__':
    try:
        test_exception()
    except Exception as e:
        print(e)