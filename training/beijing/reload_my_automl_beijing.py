import sklearn.model_selection
import pandas as pd
import sklearn.model_selection
from sklearn.metrics import mean_squared_error as mse
from joblib import load

#This code is needed so we can import the AutoML as a module
import sys
import os
sys.path.append(os.getcwd() + '/AutoML')
from automl import AutoML

#Load the data set
df_beijing = pd.read_csv("./data/beijing.csv")

#do preprocessing
df_beijing = df_beijing.drop("No", axis="columns").dropna()
y = df_beijing["pm2.5"]
X = df_beijing.loc[:, df_beijing.columns != "pm.25"]
X = X.join(pd.get_dummies(X["cbwd"])).drop("cbwd", axis="columns").drop("pm2.5", axis="columns")

#train test split
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state=123)

#reloading Auto-ML from joblib file
automl = load("./DumpedModels/beijing_automl.joblib")

#Get the final MSE of the best model
y_pred = automl.predict(X_test)
result = mse(y_test, y_pred)
print("Final model got a MSE of ", result)

