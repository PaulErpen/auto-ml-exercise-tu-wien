import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.svm import LinearSVR
from tpot.builtins import StackingEstimator
from sklearn.metrics import mean_squared_error as mse

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
df_crab = pd.read_csv("./data/crab-age.csv")

df_crab = df_crab.join(pd.get_dummies(df_crab["Sex"]))
df_crab = df_crab.drop("Sex", axis="columns")
y = df_crab.copy(deep=True)["Age"]
X = df_crab.copy(deep=True).drop("Age", axis="columns")

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=123)

# Average CV score on the training set was: -4.208671414189331
exported_pipeline = make_pipeline(
    PCA(iterated_power=10, svd_solver="randomized"),
    StackingEstimator(estimator=LinearSVR(C=20.0, dual=True, epsilon=0.1, loss="squared_epsilon_insensitive", tol=1e-05)),
    RandomForestRegressor(bootstrap=True, max_features=0.6000000000000001, min_samples_leaf=8, min_samples_split=7, n_estimators=100)
)

exported_pipeline.fit(X_train, y_train)
results = exported_pipeline.predict(X_test)
print("Got a final MSE of ", mse(y_test, results))