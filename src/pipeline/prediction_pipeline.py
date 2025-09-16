import sys
import pandas as pd
from dataclasses import dataclass
from src.exception import CustomException
from src.entity.config_entity import VehiclePredictorConfig
from src.logger import logging
from src.utils.main_utils import MainUtils


# -------------------- Input Data --------------------
class VehicleData:
    def __init__(self,
                 Gender,
                 Age,
                 Driving_License,
                 Region_Code,
                 Previously_Insured,
                 Annual_Premium,
                 Policy_Sales_Channel,
                 Vintage,
                 Vehicle_Age_lt_1_Year,
                 Vehicle_Age_gt_2_Years,
                 Vehicle_Damage_Yes):
        try:
            self.Gender = Gender
            self.Age = Age
            self.Driving_License = Driving_License
            self.Region_Code = Region_Code
            self.Previously_Insured = Previously_Insured
            self.Annual_Premium = Annual_Premium
            self.Policy_Sales_Channel = Policy_Sales_Channel
            self.Vintage = Vintage
            self.Vehicle_Age_lt_1_Year = Vehicle_Age_lt_1_Year
            self.Vehicle_Age_gt_2_Years = Vehicle_Age_gt_2_Years
            self.Vehicle_Damage_Yes = Vehicle_Damage_Yes
        except Exception as e:
            raise CustomException(e, sys)

    def get_vehicle_data_as_dict(self):
        try:
            return {
                "Gender": [self.Gender],
                "Age": [self.Age],
                "Driving_License": [self.Driving_License],
                "Region_Code": [self.Region_Code],
                "Previously_Insured": [self.Previously_Insured],
                "Annual_Premium": [self.Annual_Premium],
                "Policy_Sales_Channel": [self.Policy_Sales_Channel],
                "Vintage": [self.Vintage],
                "Vehicle_Age_lt_1_Year": [self.Vehicle_Age_lt_1_Year],
                "Vehicle_Age_gt_2_Years": [self.Vehicle_Age_gt_2_Years],
                "Vehicle_Damage_Yes": [self.Vehicle_Damage_Yes]
            }
        except Exception as e:
            raise CustomException(e, sys)

    def get_vehicle_input_data_frame(self) -> pd.DataFrame:
        try:
            return pd.DataFrame(self.get_vehicle_data_as_dict())
        except Exception as e:
            raise CustomException(e, sys)


# -------------------- Local Model Loader --------------------
class LocalModelEstimator:
    def __init__(self, model_path: str):
        try:
            self.model_path = model_path
            self.utils = MainUtils()
            self.model = self.utils.load_object(model_path)
        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, dataframe):
        try:
            return self.model.predict(dataframe)
        except Exception as e:
            raise CustomException(e, sys)


# -------------------- Prediction Pipeline --------------------
class VehicleDataClassifier:
    def __init__(self):
        try:
            self.config = VehiclePredictorConfig()
        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, dataframe) -> str:
        try:
            logging.info("Entered predict method of VehicleDataClassifier class")
            model = LocalModelEstimator(model_path=self.config.model_file_path )
            result = model.predict(dataframe)
            return result
        except Exception as e:
            raise CustomException(e, sys)



# # -------------------- Sample Usage --------------------
# if __name__ == "__main__":
#     try:
#         vehicle = VehicleData(
#             Gender="Male",
#             Age=35,
#             Driving_License=1,
#             Region_Code=28,
#             Previously_Insured=0,
#             Annual_Premium=30000,
#             Policy_Sales_Channel=152,
#             Vintage=250,
#             Vehicle_Age_lt_1_Year=0,
#             Vehicle_Age_gt_2_Years=1,
#             Vehicle_Damage_Yes=1
#         )

#         df = vehicle.get_vehicle_input_data_frame()
#         config = VehiclePredictorConfig(model_file_path="artifact/best_model/model.pkl")
#         classifier = VehicleDataClassifier(config)
#         prediction = classifier.predict(df)

#         print("Prediction:", prediction)

#     except Exception as e:
#         print("Error during prediction:", e)
