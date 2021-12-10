import autosklearn.regression
from sklearn.metrics import mean_squared_error as mse
import pandas as pd
import sklearn
from joblib import dump

df_beijing = pd.read_csv("./data/beijing.csv")

df_beijing = df_beijing.drop("No", axis="columns").dropna()
y = df_beijing["pm2.5"]
X = df_beijing.loc[:, df_beijing.columns != "pm.25"]
X = X.join(pd.get_dummies(X["cbwd"])).drop("cbwd", axis="columns")

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state=123)

autoSklearn = autosklearn.regression.AutoSklearnRegressor()
autoSklearn.fit(X_train, y_train)
y_pred = autoSklearn.predict(X_test)
print("MSE score", mse(y_test, y_pred))
dump(autoSklearn, "./DumpedModels/beijing_autosklearn.joblib")