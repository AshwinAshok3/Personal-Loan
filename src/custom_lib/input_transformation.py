'''
purpose : converts the inputs into readable for the models.

Input Features :
    - Family - Family size of the customer
        |-> integers -> (Categorical Values)

    - CCAvg = Avg. spending on credit cards per month (US Dollars)
        |-> floating point (0.0 - 10.0)-> (numeric Values)

    - Education - Education Level. 1: Undergrad; 2: Graduate; 3: Advanced/Professional
        |-> integers (1,2,3) -> (Categorical Values)

    - Income - Annual income of the customer (US Dollars)
        |-> integer val -> (numeric Values) if in 1000's will remove the zeros

    - CD Account - Does the customer have a certificate of deposit (CD) account with the bank? cat
        |-> binary values (0,1) -> (Categorical Values)

    - Mortgage - Value of house mortgage if any. (US Dollars)
        |-> integer val-> (numeric Values) if in 1000's will remove the zeros


methods that we will use :
*   convert these inputs and then return the column transformers
*   here we have 6 features,
    - so after standardization for numerics and One Hot Encoding for categorical values
* we will be having 12 feature extracted after everything.
'''


# IMPORT LIBRARIES
# ================================================================== #
# importing loggers for tracking the error logs and finished tasks and exceptions
from src.logger import logger
from src.exception import CustomException
import sys

# importing column transformer
from sklearn.compose import ColumnTransformer

# importing metrics like standardization and encoding
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pandas as pd


# =================================================================== #
# CLASS for Validating INPUTS
# =================================================================== #
# class for taking inputs
class FeatureValidator:
    def __init__(self, feature_dict: dict):
        self.feature_dict = feature_dict

    logger.info("Initiating Family Input ...")
    def family_input(self):
        fam_input = self.feature_dict.get("Family")
        try:
            if isinstance(fam_input, int):
                logger.info("Family Input is valid")
                return fam_input
        except Exception as e:
            logger.error("Family Input is not valid, \n must be an integer and income without 000's ")
            raise CustomException(e, sys)


    logger.info("Initiating CCAvg [Credit Card Spending Per Month] Input ...")
    def ccavg_input(self):
        ccvg = self.feature_dict.get("CCAvg")
        try :
            if isinstance(ccvg, float) and 0.0 <= ccvg <= 10.0:
                logger.info("CCAvg Input Valid")
                return ccvg
        except Exception as e:
            logger.error("Invalid Input for CCAvg, must be a float between 0.0 and 10.0.")
            raise CustomException(e, sys)

    logger.info("Initiating Education Input...")
    def education_input(self):
        edu = self.feature_dict.get("Education")
        try:
            if isinstance(edu, int) and 1 <= edu <= 3:
                logger.info("Education Input Valid")
                return edu
        except Exception as e:
            logger.error("Education Input is Invalid")
            raise CustomException(e, sys)

    logger.info("Initiating Income Input ...")
    def income_input(self):
        income = self.feature_dict.get("Income")
        try:
            if isinstance(income, int):
                logger.info("Income Input Valid")
                return income
        except Exception as e:
            logger.error("Income Input Invalid")
            raise CustomException(e, sys)


    logger.info("Initiating CD Account Input ...")
    def cdACC_input(self):
        cd_acc = self.feature_dict.get("CD Account")
        try:
            if isinstance(cd_acc, int) and cd_acc in [0, 1]:
                logger.info("CD Account Input Valid")
                return cd_acc
        except Exception as e:
            logger.error("Either not in range or Not an integer")
            raise CustomException(e, sys)

    logger.info("Initiating Mortgage Input ...")
    def mortgage_input(self):
        mrtg = self.feature_dict.get("Mortgage")
        try:
            if isinstance(mrtg, int):
                logger.info("Mortgage Input Valid")
                return mrtg
        except Exception as e:
            logger.error("Mortgage not INT: Invalid")
            raise CustomException(e, sys)

    def get_validated_inputs(self):
        logger.info("Valid INPUT Ready for Transformation !")
        return {
            "Family": self.family_input(),
            "CCAvg": self.ccavg_input(),
            "Education": self.education_input(),
            "Income": self.income_input(),
            "CD Account": self.cdACC_input(),
            "Mortgage": self.mortgage_input()
        }



# ==================================================================== #
#  CLASS FOR TRANSFORMING THE INPUT
# ==================================================================== #
class FeatureTransformer:
    def data_transform(self, input_data: dict):
        df = pd.DataFrame([input_data])
        numerical_cols = ["CCAvg", "Income", "Mortgage"]
        categorical_cols = ["Family", "Education", "CD Account"]
        preprocessing = ColumnTransformer(
            transformers=[
                ('numericals', StandardScaler(), numerical_cols),
                ('categorical', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
            ]
        )
        return preprocessing.fit_transform(df)

# Main execution
if __name__ == "__main__":
    logger.info("fetching Inputs ...")
    user_input_dict = {
        "Family": 3,
        "CCAvg": 2.5,
        "Education": 2,
        "Income": 50000,
        "CD Account": 1,
        "Mortgage": 100000
    }

    try:
        logger.info("Validating Given Inputs")
        validator = FeatureValidator(user_input_dict)
        validated_data = validator.get_validated_inputs()
        print("Validated Input Data:", validated_data)

        transformer = FeatureTransformer()
        transformed_data = transformer.data_transform(validated_data)
        print("\nTransformed Data (Ready for Model):\n", transformed_data)

    except Exception as e:
        print("\nValidation Error:", e)
