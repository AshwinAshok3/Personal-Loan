############################################
#             IMPORTING LIBRARIES          #
############################################
# data cleaning tools
import pandas as pd

# ml tools
from catboost import CatBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

# neural network tools
import tensorflow as tf

# data transformation tools
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

# feature engineering tools
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest, mutual_info_classif

# metrics tools
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix

# model saving tool
import joblib

# logs and exceptions
from src.logger import logger
from src.exception import CustomException
import sys


#################################################
#              DATA CLEANING                    #
#################################################
try :
    # importing the dataset
    df = pd.read_csv('src/data/cleaned_data.csv', index_col=0)
    logger.info("Imported dataset")

    # deleting duplicated rows
    df.drop_duplicates(inplace=True)
    logger.info("Dropping duplicate columns")

    # dropping null values if exists
    df.dropna(inplace=True, axis=0)
    logger.info("Dropping missing or null values")
    
    # dropping the "ZIP Code" column
    df.drop(columns={"ZIP Code"}, axis=1, inplace = True)
    
except Exception as e:
    raise CustomException(e, sys)

# assigning columns to a variable
df_cols = list(df.columns)

pd.set_option("display.max_columns",None)
df.head(5)

# displaying uniques values for each columns
for i in df_cols:
    print(f"Column : {i}")
    print(df[i].unique())
    print("*"*20)

##################################################
#               DATA TRANSFORMATION              #
##################################################
if "Personal Loan" in df_cols:
    df_cols.remove("Personal Loan")
# print(df_cols)

# dividing the target and independent variable
X = df[df_cols]
y = df["Personal Loan"]
logger.info("Separated X & y")

# numerical columns
num_col = ["Age", "Experience", "Income", "CCAvg", "Mortgage"]
print(num_col)

# initiating the transformer
cols_transform = ColumnTransformer(
    transformers=[("Numbers", StandardScaler(), num_col)],
    remainder='passthrough'
    )

# fit and transform the train set onto the transfomer
X_transformed = cols_transform.fit_transform(X)
logger.info("Transformed 'X' data")

# saving the column transformer model
joblib.dump(cols_transform, "models/column_transformer.pkl")

##################################################
#                 SPLIT THE DATA                 #
##################################################

# splitting train and test dataset with equal proportion of 'y'
X_train , X_test , y_train , y_test = train_test_split(X_transformed, y, 
                                                       test_size=0.3, 
                                                       random_state=3, 
                                                       stratify=y)
logger.info("Splitted X & y into train & test data")

print(f"X_Train set shape {X_train.shape}")
print(f"y_Train set shape {y_train.shape}")
print(f"X_Test set shape {X_test.shape}")
print(f"y_Test set shape {y_test.shape}")

##################################################
#------------- OVER SAMPLING THE DATA -----------#
##################################################
# over sampling the train set which has minority
# initializing the SMOTE func
smt = SMOTE(sampling_strategy='auto')

# resampling the X_train
X_train_resampled, y_train_resampled = smt.fit_resample(X_train, y_train)
logger.info("Resampled the minority data on the train set")

# shape of the resampled dataset
print(f"X_Train_resampled set shape \n X_train : {X_train_resampled.shape} \n \
      X_test : {y_train_resampled.shape}")


###################################################
#--------------- FEATURE SELECTION ---------------#
###################################################

# initializing the SelectKBest with mutual classif info for 5 top features
feature_selector = SelectKBest(mutual_info_classif, k=6)
logger.info("SelectKBest and mutual_info_classif initiated, implemented!!")

# fitting the train and selecting the best 5 features from the train set
X_train_selected = feature_selector.fit_transform(X_train_resampled, y_train_resampled)

# transforming the test set
X_test_selected = feature_selector.transform(X_test)
logger.info("Best features Choosen")

print(f"X_Train set shape new: {X_train_selected.shape}")
print(f"y_Train set shape new {y_train_resampled.shape}")
print(f"X_Test set shape {X_test_selected.shape}")
print(f"y_Test set shape {y_test.shape}")

# Get boolean mask of selected features
mask = feature_selector.get_support()

# Use it to filter original column names
selected_features = X.columns[mask]
print("Choosen Features are :")
print(selected_features.tolist())  # Convert to list if needed

# saving the train set and test set
# train set
# Recombine selected features and target into one DataFrame
train_set = pd.DataFrame(X_train_selected, columns=selected_features)
train_set['target'] = y_train_resampled  # Append target column

# Save to CSV
train_set.to_csv("data/train.csv", index=False)

# test set
test_set = pd.DataFrame(X_test_selected, columns=selected_features)
test_set['target'] = y_test  # Append target column
test_set.to_csv("data/test.csv", index=False)

###################################################
#----------------- MODEL CHOOSING ----------------#only using training data
###################################################
# catboost parameters
catboost_params = {
    'iterations': [300, 600, 900],               # Controls model complexity, keep low to moderate
    'depth': [4, 6, 8],                          # Tree depth: deeper = more complex
    'learning_rate': [0.05, 0.1],                # Smaller = slower but more precise
    'l2_leaf_reg': [3, 7],                       # Regularization
    'random_strength': [1, 2],                   # Controls randomness of splits (like regularization)
    'bagging_temperature': [0.5, 1.0],           #
    'scale_pos_weight': [1, 2, 3]                #
}

# gradient boost parameters
gradient_boost_params = {
   'n_estimators': [300, 600, 900],           # Number of boosting stages
   'learning_rate': [0.05, 0.1],              # Step size shrinkage
   'max_depth': [3, 5, 7],                    # Max depth of individual trees
   'min_samples_split': [2, 5],               # Minimum samples to split
   'min_samples_leaf': [1, 3],                # Minimum samples at leaf
   'subsample': [0.5, 0.8]                    # Fraction of samples used for fitting each tree
    }

# Initiating the Catboost Classifier upon gridsearch CV
catboost_grid = GridSearchCV(CatBoostClassifier(), param_grid = catboost_params, 
                             cv = 3, scoring='f1', n_jobs=-1)
logger.info("Initiated Catboost Classifier ")

# initiating gradient boosting classifier on gridsearchcv
gradboost_grid = GridSearchCV(GradientBoostingClassifier(), param_grid = gradient_boost_params,
                              cv=3, scoring='f1', n_jobs=-1)
logger.info("Initiated Gradient Boosting Classifier")


# fitting the train set on catboost Classifier
logger.info("started Catboost hyperparameter Tuning")
catboost_grid.fit(X_train_selected, y_train_resampled)
logger.info("Fitting data on Cat Boosting Classifier [Parameter Tuning Finished]")


# fitting the train set on gradient boosting classifier
logger.info("Started Gradient Boosting Classifier hyperparameter tuning")
gradboost_grid.fit(X_train_selected, y_train_resampled)
logger.info("Fitting data on Gradient Boosting Classifier [Parameter Tuning Finished]")


# saving the final results for catboost
catboost_result_df = pd.DataFrame.from_dict(catboost_grid.cv_results_)
catboost_result_df.to_csv("models/catboost_results.csv")
logger.info("CatBoost Model Saved")

# saving the final results for catboost
gradboost_result_df = pd.DataFrame.from_dict(gradboost_grid.cv_results_)
gradboost_result_df.to_csv("models/gradboost_results.csv")
logger.info("Gradient Boost Model Saved")


#  for catboost results
print(catboost_result_df[catboost_result_df['rank_test_score']==1])

# grad boost results
print(gradboost_result_df[gradboost_result_df['rank_test_score']==1])

###################################################
#----------------- MODEL FITTING -----------------# only using training data
###################################################

# gradient boosting model's best performance
grad_boost_best_params = gradboost_grid.best_params_
logger.info("Getting Best parameters")

# fitting best parameters with training set
final_ml_model = GradientBoostingClassifier(**grad_boost_best_params)
final_ml_model.fit(X_train_selected, y_train_resampled)
logger.info("Initiated Gradient Boosting Classifier and fitted the model !")

###################################################
#--------- EVALUATE MODEL PERFORMANCE ------------# evaluating test data
###################################################

# grad boost evaluation on test set
y_pred = final_ml_model.predict(X_test_selected)
logger.info("Prediction testing started")

# Evaluating metrics
acc_score = accuracy_score(y_test, y_pred)
f1_score_ = f1_score(y_test, y_pred)
roc_score_auc = roc_auc_score(y_test, y_pred)
confusion_matrix1 = confusion_matrix(y_test, y_pred)
classification_report1 = classification_report(y_test, y_pred)

logger.info("Evaluation metric started")
# output all the metrics evaluation
print(f"Accuracy Score : {acc_score}")
print(f"F1 Score : {f1_score_}")
print(f"ROC AUC Score : {roc_score_auc}")
print("*"*20)
print(f"Confusion Matrix : \n{confusion_matrix1}")
print("*"*20)
print(f"Classification Report: \n{classification_report1}")

joblib.dump(final_ml_model, "models/gradientboostC_model.pkl")
logger.info("ML Model Saved")


###################################################
# ---------------- NEURAL NETWORK ----------------#
###################################################
logger.info("INitiated Neural Network")
# initializing the selector for tensorflow
nn_model = tf.keras.Sequential()

# validating inputs layer
nn_model.add(tf.keras.Input(shape=(len(selected_features.tolist()),)))
logger.info("Initiated Neural Network and input shape done for data")

# first layer of the neural network
nn_model.add(tf.keras.layers.Dense(units = 16, activation='relu'))
logger.info("First layer added to the neural network")

# second layer
nn_model.add(tf.keras.layers.Dense(units = 128, activation='swish'))
logger.info("Second layer added to the neural network")

# adding a dropout layer
nn_model.add(tf.keras.layers.Dropout(0.35))
logger.info("Dropout layer added to the network")

# third layer
nn_model.add(tf.keras.layers.Dense(units = 64, activation='leaky_relu'))
logger.info("Third layer added to the neural network")

# final output layer
nn_model.add(tf.keras.layers.Dense(units = 1, activation='sigmoid'))
logger.info("output layer added to the network")

# Neural network compiling
nn_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy',tf.keras.metrics.AUC(name='auc')]
)
logger.info("Compiling the neural network")

# fitting the neural network model
nn_model.fit(
    X_train_selected, y_train_resampled,
    validation_data=(X_test_selected, y_test),  # optional
    epochs=100,
    batch_size=100,
    verbose=1
)
logger.info("Fit & training the model")

# fitting the neural network model
nn_model.fit(
    X_train_selected, y_train_resampled,
    validation_data=(X_test_selected, y_test),  # optional
    epochs=100,
    batch_size=100,
    verbose=2
)
logger.info("Fit & training the model")

# saving the nn_model into keras format
nn_model.save("models/neural_net.keras")
logger.info("NN Model saved")

#########################
#       END
##########################












