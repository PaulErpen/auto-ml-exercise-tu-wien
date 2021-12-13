import autosklearn.regression
import sklearn
from sklearn.metrics import mean_squared_error as mse
import pandas as pd
from joblib import load

df_fifa = pd.read_csv("./data/fifa_preprocessed.csv")

y = df_fifa.copy(deep=True)["Value"]
X = df_fifa.copy(deep=True).drop("Value", axis='columns')

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state=123)

autoSklearn = load("./DumpedModels/fifa_autosklearn.joblib")
y_pred = autoSklearn.predict(X_test)
print("MSE score", mse(y_test, y_pred))