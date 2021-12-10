import autosklearn.regression
import sklearn.model_selection
import pandas as pd
from joblib import dump

df_beijing = pd.read_csv("../../data/beijing.csv")

df_beijing = df_beijing.drop("No", axis="columns").dropna()
y = df_beijing["pm2.5"]
X = df_beijing.loc[:, df_beijing.columns != "pm.25"]
X = X.join(pd.get_dummies(X["cbwd"])).drop("cbwd", axis="columns")

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state=1)

autoSklearn = autosklearn.regression.AutoSklearnRegressor()
autoSklearn.fit(X_train, y_train)
y_hat = autoSklearn.predict(X_test)
print("Accuracy score", sklearn.metrics.accuracy_score(y_test, y_hat))
dump(autoSklearn, "./DumpedModels/beijing_autosklearn.joblib")