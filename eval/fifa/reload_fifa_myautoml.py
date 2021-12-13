from sklearn.metrics import mean_squared_error as mse
import pandas as pd
import sklearn
from joblib import load
import sys
import os
sys.path.append(os.getcwd() + '/AutoML')
from automl import AutoML

df_fifa = pd.read_csv("./data/fifa_final.csv")

y = df_fifa.copy(deep=True)["Value"]
X = df_fifa.copy(deep=True).drop("Value", axis='columns')

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state=123)

automl = load("./DumpedModels/fifa_automl.joblib")
y_pred = automl.predict(X_test)
print("MSE score ", mse(y_test, y_pred), " with the model ", automl.get_best_solution_algorithm_name())
