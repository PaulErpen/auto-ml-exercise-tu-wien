import sklearn.model_selection
import pandas as pd
import sklearn.model_selection
from sklearn.metrics import mean_squared_error as mse
from joblib import load
import autosklearn.regression

df_beijing = pd.read_csv("./data/beijing.csv")

df_beijing = df_beijing.drop("No", axis="columns").dropna()
y = df_beijing["pm2.5"]
X = df_beijing.loc[:, df_beijing.columns != "pm.25"]
X = X.join(pd.get_dummies(X["cbwd"])).drop("cbwd", axis="columns")

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state=123)

#reloading Auto-ML from joblib file
autosklearn_reload = load("./DumpedModels/beijing_autosklearn.joblib")

#Get the final MSE of the best model
y_pred = autosklearn_reload.predict(X_test)
result = mse(y_test, y_pred)
print("Final model got a MSE of ", result)
print(autosklearn_reload.get_models_with_weights())