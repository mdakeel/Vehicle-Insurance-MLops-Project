import os
import sys

from pandas import DataFrame
from src.entity.config_entity import DataIngestionConfig
from sklearn.model_selection import train_test_split
from src.entity.artifact_entity import DataIngestionArtifact
from src.exception import CustomException
from src.logger import logging
from src.data_access.proj1_data import Proj1Data

class DataIngestion:
    def __init__(self):
        try:
            self.config = DataIngestionConfig()
        except Exception as e:
            raise CustomException(e,sys)
        

    def export_data_into_feature_store(self)->DataFrame:
        try:
            logging.info(f"Exporting data from mongodb")
            my_data = Proj1Data()
            dataframe = my_data.export_collection_as_dataframe(collection_name=self.config.collection_name)
            logging.info(f"Shape of dataframe: {dataframe.shape}")
            os.makedirs(os.path.dirname(self.config.row_data_path), exist_ok=True)
            dataframe.to_csv(self.config.row_data_path, index=False, header=True)
            logging.info(f"Saved raw data to: {self.config.row_data_path}")
            return dataframe
        except Exception as e:
            raise CustomException(e,sys)
        
    def split_data_as_train_test(self, dataframe: DataFrame) -> None:
        try:
            logging.info("Splitting data into train and test sets")
            train_df, test_df = train_test_split(dataframe, test_size=self.config.train_test_split_ratio, random_state=42)
            
            dir_path = os.path.dirname(self.config.train_file_name)
            os.makedirs(dir_path,exist_ok=True)
            
            logging.info(f"Exporting train and test file path.")
            
            train_df.to_csv(self.config.train_file_name, index=False, header=True)
            test_df.to_csv(self.config.test_file_name, index=False, header=True)
            logging.info("Train and test data saved successfully")
        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logging.info("Initiating data ingestion pipeline")
            df = self.export_data_into_feature_store()
            self.split_data_as_train_test(df)

            artifact = DataIngestionArtifact(
                trained_file_path=self.config.train_file_name,
                test_file_path=self.config.test_file_name
            )

            logging.info(f"Data ingestion artifact created: {artifact}")
            return artifact

        except Exception as e:
            raise CustomException(e, sys)