import autosklearn.regression
import sklearn
from sklearn.metrics import mean_squared_error as mse
import pandas as pd
from joblib import dump

df_crab = pd.read_csv("./data/crab-age.csv")

df_crab = df_crab.join(pd.get_dummies(df_crab["Sex"]))
df_crab = df_crab.drop("Sex", axis="columns")
y = df_crab.copy(deep=True)["Age"]
X = df_crab.copy(deep=True).drop("Age", axis="columns")

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state=1)

autoSklearn = autosklearn.regression.AutoSklearnRegressor()
autoSklearn.fit(X_train, y_train)
y_pred = autoSklearn.predict(X_test)
print("MSE score", mse(y_test, y_pred))
dump(autoSklearn, "./DumpedModels/beijing_autosklearn.joblib")