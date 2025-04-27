'''
Main aim for this python file is try and fit the given input
'''

# libraries to get functions from class
from src.custom_lib.input_transformation import FeatureValidator, FeatureTransformer
from src.exception import CustomException
from src.logger import logger

# importing required libraries for python functions and libs
import tensorflow as tf
import sys
import joblib
import numpy

################################################################
######### Fitting the Inputs into class and predicting #########
################################################################
class fit_input:
    def model_init(inputs):
        # initializing the models
        catboost_model = joblib.load("src/models/final_model/personal_loan_ML_catbst_model.pkl")
        gradboost_model = joblib.load("src/models/final_model/personal_loan_ML_grad_boost_model.pkl")
        # nn_model1 = load_model("src/models/final_model/personal_loan_NN_model.h5")
        nn_model2 = tf.keras.models.load_model("src/models/final_model/personal_loan_NN_model1.keras")
        nn_model3 = tf.keras.models.load_model("src/models/final_model/personal_loan_NN_model2.keras")

        try:
            logger.info("Fitting Catboost ..")
            cat_model_output = catboost_model.predict(inputs)
            logger.info("Finished fitting Cat Boost ..")

            logger.info("Fitting Gradient Boosting ..")
            gradboost_output = gradboost_model.predict(inputs)
            logger.info("Finished fitting Gradient Boosting ..")

            # logger.info("Fitting NN_Model #No.1 ..")
            # nn_model_output1 = nn_model1.predict(inputs)
            # logger.info("Finished fitting NN_Model #No.1 ..")

            logger.info("Fitting NN_Model #No.2 ..")
            nn_model_output2 = nn_model2.predict(inputs)
            logger.info("Finished fitting NN_Model #No.2 ..")

            logger.info("Fitting NN_Model #No.3 ..")
            nn_model_output3 = nn_model3.predict(inputs)
            logger.info("Finished fitting NN_Model #No.3 ..")
        except Exception as e:
            raise CustomException(e, sys)

        logger.info("Finished fitting all models...")

        models_prediction = {
            "Catboost": cat_model_output,
            "Gradient Boost": gradboost_output,
            # "NN_Model": nn_model_output1,
            "NN_Model2": nn_model_output2,
            "NN_Model3": nn_model_output3
        }

        return models_prediction





# Main execution
if __name__ == "__main__":
    print(tf.__version__)

    logger.info("Inputs Fetched...")
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

        logger.info("Initiating Transformer...")
        transformer = FeatureTransformer()

        logger.info("Transforming Input ...")
        data_transformed = transformer.data_transform(validated_data)

        logger.info("Finished Transform")
        print("\nTransformed Data (Ready for Model):\n", data_transformed)

        logger.info("Trying to fit models....")
        model_initializer = fit_input.model_init(data_transformed)

    except Exception as e:
        print("\nValidation Error:", e)
        raise CustomException(e, sys)