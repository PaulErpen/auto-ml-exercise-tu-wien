from sklearn.metrics import mean_squared_error as mse
import pandas as pd
import sklearn
from joblib import load
import sys
import os
sys.path.append(os.getcwd() + '/AutoML')
from automl import AutoML

df_crab = pd.read_csv("./data/crab-age.csv")

df_crab = df_crab.join(pd.get_dummies(df_crab["Sex"]))
df_crab = df_crab.drop("Sex", axis="columns")
y = df_crab.copy(deep=True)["Age"]
X = df_crab.copy(deep=True).drop("Age", axis="columns")

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state=123)

automl = load("./DumpedModels/crab-age_automl.joblib")
y_pred = automl.predict(X_test)
print("MSE score", mse(y_test, y_pred), " with model of type ", automl.get_best_solution_algorithm_name())
