from sklearn.metrics import mean_squared_error as mse
import pandas as pd
from joblib import dump
from tpot import TPOTRegressor
import sklearn

df_fifa = pd.read_csv("./data/fifa_preprocessed.csv")

y = df_fifa.copy(deep=True)["Value"]
X = df_fifa.copy(deep=True).drop("Value", axis='columns')

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state=123)

tpot = TPOTRegressor(generations=None, max_time_mins=60)
tpot.fit(X_train, y_train)
y_pred = tpot.predict(X_test)
print("MSE score", mse(y_test, y_pred))
tpot.export("./DumpedModels/fifa_tpot.py")