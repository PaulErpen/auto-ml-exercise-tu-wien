from sklearn.metrics import mean_squared_error as mse
import pandas as pd
import sklearn
from joblib import dump
import sys
import os
sys.path.append(os.getcwd() + '/AutoML')
from automl import AutoML

df_fifa = pd.read_csv("./data/fifa_preprocessed.csv")

y = df_fifa.copy(deep=True)["Value"]
X = df_fifa.copy(deep=True).drop("Value", axis='columns')

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state=123)

automl = AutoML(logging_enabled=True, runtime_seconds=60*60, csv_output_enabled=True, csv_output_folder="./output/fifa")
automl.fit(X_train, y_train)
y_pred = automl.predict(X_test)
print("MSE score", mse(y_test, y_pred))
dump(automl, "./DumpedModels/fifa_automl.joblib")