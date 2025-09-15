import sys
import os
import pickle
import yaml
import pandas as pd
import numpy as np
from typing import Dict, Tuple
from src.constants import *
from src.exception import CustomException
from src.logger import logging

class MainUtils:
    def __init__(self) -> None:
        pass
    
    def read_yaml_file(file_path: str) -> dict:
        try:
            with open(file_path, "rb") as yaml_file:
                return yaml.safe_load(yaml_file)
    
        except Exception as e:
            raise CustomException(e, sys) from e
    
    
    def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
        try:
            if replace:
                if os.path.exists(file_path):
                    os.remove(file_path)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "w") as file:
                yaml.dump(content, file)
        except Exception as e:
            raise CustomException(e, sys) from e
    
    
    # def load_object(file_path: str) -> object:
    #     """
    #     Returns model/object from project directory.
    #     file_path: str location of file to load
    #     return: Model/Obj
    #     """
    #     try:
    #         with open(file_path, "rb") as file_obj:
    #             obj = dill.load(file_obj)
    #         return obj
    #     except Exception as e:
    #         raise CustomException(e, sys) from e
    
    def save_numpy_array_data(self, file_path: str, array: np.array):
        """
        Save numpy array data to file
        file_path: str location of file to save
        array: np.array data to save
        """
        try:
            dir_path = os.path.dirname(file_path)
            os.makedirs(dir_path, exist_ok=True)
            with open(file_path, 'wb') as file_obj:
                np.save(file_obj, array)
        except Exception as e:
            raise CustomException(e, sys) from e
    
    
    def load_numpy_array_data(self, file_path: str) -> np.array:
        """
        load numpy array data from file
        file_path: str location of file to load
        return: np.array data loaded
        """
        try:
            with open(file_path, 'rb') as file_obj:
                return np.load(file_obj)
        except Exception as e:
            raise CustomException(e, sys) from e
    
    
    # def drop_columns(df: DataFrame, cols: list)-> DataFrame:
    
    #     """
    #     drop the columns form a pandas DataFrame
    #     df: pandas DataFrame
    #     cols: list of columns to be dropped
    #     """
    #     logging.info("Entered drop_columns methon of utils")
    
    #     try:
    #         df = df.drop(columns=cols, axis=1)
    
    #         logging.info("Exited the drop_columns method of utils")
            
    #         return df
    #     except Exception as e:
    #         raise CustomException(e, sys) from e
        
    @staticmethod
    def save_object(file_path: str, obj: object) -> None:
        """Save any Python object to a file using pickle."""
        logging.info(f'Saving object to {file_path}')
        
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'wb') as file_obj:
                pickle.dump(obj, file_obj)
            logging.info(f'Successfully saved object to {file_path}')
        except Exception as e:
            logging.error(f'Faild to save object to {file_path}')
            raise CustomException(e, sys) from e
    
    @staticmethod
    def load_object(file_path: str) -> None:
        """Load any Pickled Python object from a file."""
        logging.info(f'Loading object from {file_path}')
        
        try:
            with open(file_path, 'rb') as file_obj:
                obj = pickle.load(file_obj)
            logging.info(f'Successfully loaded object from {file_path}')
            return obj
        except Exception as e:
            logging.error(f'Faild to load object from {file_path}')
            raise CustomException(e, sys) from e
    
   
         