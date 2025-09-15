import os
import sys
import json
import pandas as pd
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
from src.utils.main_utils import MainUtils
from src.entity.config_entity import DataValidationConfig
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from src.constants import SCHEMA_FILE_PATH

@dataclass
class DataValidation:
    def __init__(self, artifact: DataIngestionArtifact, config: DataValidationConfig):
        self.config = config
        self.artifact = artifact
        self.schema = MainUtils.read_yaml_file(SCHEMA_FILE_PATH)

    def _validate_columns(self, df: pd.DataFrame) -> bool:
        status = len(df.columns) == len(self.schema["columns"])
        logging.info(f"Is required column present: [{status}]")
        return status

    def _check_required_columns(self, df: pd.DataFrame) -> bool:
        missing = []
        for col in self.schema["numerical_columns"] + self.schema["categorical_columns"]:
            if col not in df.columns:
                missing.append(col)
        if missing:
            logging.info(f"Missing columns: {missing}")
        return len(missing) == 0
    
    def _read_csv(self, path: str) -> pd.DataFrame:
        try:
            return pd.read_csv(path)
        except Exception as e:
            raise CustomException(e, sys)

    def validate(self) -> DataValidationArtifact:
        try:
            logging.info("Starting data validation...")

            train_df = self._read_csv(self.artifact.trained_file_path)
            test_df = self._read_csv(self.artifact.test_file_path)

            errors = []

            if not self._validate_columns(train_df):
                errors.append("Training data column count mismatch.")
            if not self._validate_columns(test_df):
                errors.append("Test data column count mismatch.")

            if not self._check_required_columns(train_df):
                errors.append("Training data missing required columns.")
            if not self._check_required_columns(test_df):
                errors.append("Test data missing required columns.")

            status = len(errors) == 0
            report = {
                "validation_status": status,
                "message": " | ".join(errors) if errors else "Validation passed."
            }

            os.makedirs(os.path.dirname(self.config.validation_report_file_path), exist_ok=True)
            with open(self.config.validation_report_file_path, "w") as f:
                json.dump(report, f, indent=4)

            logging.info("Validation report saved.")
            return DataValidationArtifact(
                validation_status=status,
                message=report["message"],
                validation_report_file_path=self.config.validation_report_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)
