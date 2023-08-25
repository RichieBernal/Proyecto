import logging
import pandas as pd
import numpy as np
import re
import os

logger = logging.getLogger(__name__) # Indicamos que tome el nombre del modulo
logger.setLevel(logging.DEBUG) # Configuramos el nivel de logging

formatter = logging.Formatter('%(asctime)s:%(name)s:%(module)s:%(levelname)s:%(message)s') # Creamos el formato

file_handler = logging.FileHandler('FAE\logs\load.log') # Indicamos el nombre del archivo

file_handler.setFormatter(formatter) # Configuramos el formato

logger.addHandler(file_handler) # Agregamos el archivo

class DataRetriever:
    """
    A class for retrieving data from a given A data path and processing it for further analysis.

    Parameters:
        data_path (str): The file path from which the data will be loaded.

    Attributes:
        data_path (str): The file path from which the data will be loaded.
        """
    

    RETRIEVED_DATA = 'data_fire.csv' # File name for the retrieved data.

    def __init__(self, url, data_path):
        self.url = url
        self.DATASETS_DIR = data_path

    def retrieve_data(self):
        """
        Retrieves data from the specified path, processes it, and stores it in a CSV file.

        Returns:
            str: A message indicating the location of the stored data.
        """    
        # Loading data from specific path
        data = pd.read_csv(self.url) 

        # Create directory if it does not exist
        if not os.path.exists(self.DATASETS_DIR):
            os.makedirs(self.DATASETS_DIR)
            logger.info("Directory create successfully")
            print(f"Directory '{self.DATASETS_DIR}' created successfully.")
        else:
            logger.info("Directory alraeady successfully")
            print(f"Directory '{self.DATASETS_DIR}' already exists.")

        # Save data to CSV file
        data.to_csv(self.DATASETS_DIR + self.RETRIEVED_DATA, index=False)

        logger.info("Data saved")
        return f'Data stored in {self.DATASETS_DIR + self.RETRIEVED_DATA}'
    




