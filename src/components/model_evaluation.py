import sys
import pandas as pd
from typing import Optional
from dataclasses import dataclass
from sklearn.metrics import f1_score

from src.entity.config_entity import ModelEvaluationConfig
from src.entity.artifact_entity import ModelTrainerArtifact, DataIngestionArtifact, ModelEvaluationArtifact
from src.exception import CustomException
from src.constants import TARGET_COLUMN
from src.logger import logging
from src.utils.main_utils import MainUtils


@dataclass
class EvaluateModelResponse:
    trained_model_f1_score: float
    best_model_f1_score: Optional[float]
    is_model_accepted: bool
    difference: float


class ModelEvaluation:

    def __init__(self, model_eval_config: ModelEvaluationConfig,
                 data_ingestion_artifact: DataIngestionArtifact,
                 model_trainer_artifact: ModelTrainerArtifact):
        try:
            self.model_eval_config = model_eval_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.model_trainer_artifact = model_trainer_artifact
            self.utils = MainUtils()
        except Exception as e:
            raise CustomException(e, sys)

    def load_best_model(self) -> Optional[object]:
        """Load best model from local path if available."""
        try:
            best_model_path = self.model_eval_config.best_model_path
            return self.utils.load_object(best_model_path)
        except Exception as e:
            logging.warning(f"Best model not found at {self.model_eval_config.best_model_path}")
            return None  # Gracefully handle missing model

    def _map_gender_column(self, df):
        logging.info("Mapping 'Gender' column to binary values")
        df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1}).astype(int)
        return df

    def _create_dummy_columns(self, df):
        logging.info("Creating dummy variables for categorical features")
        df = pd.get_dummies(df, drop_first=True)
        return df

    def _rename_columns(self, df):
        logging.info("Renaming specific columns and casting to int")
        df = df.rename(columns={
            "Vehicle_Age_< 1 Year": "Vehicle_Age_lt_1_Year",
            "Vehicle_Age_> 2 Years": "Vehicle_Age_gt_2_Years"
        })
        for col in ["Vehicle_Age_lt_1_Year", "Vehicle_Age_gt_2_Years", "Vehicle_Damage_Yes"]:
            if col in df.columns:
                df[col] = df[col].astype('int')
        return df

    def _drop_id_column(self, df):
        logging.info("Dropping 'id' column")
        if "_id" in df.columns:
            df = df.drop("_id", axis=1)
        return df

    def evaluate_model(self) -> EvaluateModelResponse:
        try:
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            x, y = test_df.drop(TARGET_COLUMN, axis=1), test_df[TARGET_COLUMN]

            logging.info("Transforming test data for prediction...")
            x = self._map_gender_column(x)
            x = self._drop_id_column(x)
            x = self._create_dummy_columns(x)
            x = self._rename_columns(x)

            trained_model = self.utils.load_object(self.model_trainer_artifact.trained_model_file_path)
            trained_model_f1_score = self.model_trainer_artifact.metric_artifact.f1_score
            logging.info(f"F1 Score (Trained Model): {trained_model_f1_score}")

            best_model = self.load_best_model()
            best_model_f1_score = None

            if best_model is not None:
                y_hat_best = best_model.predict(x)
                best_model_f1_score = f1_score(y, y_hat_best)
                logging.info(f"F1 Score (Best Model): {best_model_f1_score}")

            tmp_best_score = 0 if best_model_f1_score is None else best_model_f1_score
            is_model_accepted = trained_model_f1_score > tmp_best_score

            return EvaluateModelResponse(
                trained_model_f1_score=trained_model_f1_score,
                best_model_f1_score=best_model_f1_score,
                is_model_accepted=is_model_accepted,
                difference=trained_model_f1_score - tmp_best_score
            )

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        try:
            logging.info("Starting model evaluation...")
            evaluation_result = self.evaluate_model()
            best_model_path = self.model_eval_config.best_model_path

            if evaluation_result.is_model_accepted:
                trained_model = self.utils.load_object(self.model_trainer_artifact.trained_model_file_path)
                self.utils.save_object(best_model_path, trained_model)
                logging.info("New model accepted and saved as best model locally.")

            return ModelEvaluationArtifact(
                is_model_accepted=evaluation_result.is_model_accepted,
                trained_model_path=self.model_trainer_artifact.trained_model_file_path,
                best_model_path=best_model_path,
                changed_accuracy=evaluation_result.difference
            )

        except Exception as e:
            raise CustomException(e, sys)
