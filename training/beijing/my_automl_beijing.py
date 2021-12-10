import sklearn.model_selection
import pandas as pd
import sklearn.model_selection
from sklearn.metrics import mean_squared_error as mse
from joblib import dump
import sys
import os
sys.path.append(os.getcwd() + '/AutoML')
from automl import AutoML

df_beijing = pd.read_csv("./data/beijing.csv")

df_beijing = df_beijing.drop("No", axis="columns").dropna()
y = df_beijing["pm2.5"]
X = df_beijing.loc[:, df_beijing.columns != "pm.25"]
X = X.join(pd.get_dummies(X["cbwd"])).drop("cbwd", axis="columns").drop("pm2.5", axis="columns")

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state=None)
automl = AutoML(logging_enabled=True, max_runtime_seconds=60*60, csv_output_enabled=True, csv_output_folder="./output/beijing")
automl.fit(X_train, y_train)
y_pred = automl.predict(X_test)
y_pred = automl.predict(X_test)
result = mse(y_test, y_pred)
print("Final model got a MSE of ", result)
dump(automl, "./DumpedModels/beijing_automl.joblib")