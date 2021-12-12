import numpy as np
import pandas as pd
from sklearn.linear_model import LassoLarsCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error as mse

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
df_beijing = pd.read_csv("./data/beijing.csv")

df_beijing = df_beijing.drop("No", axis="columns").dropna()
y = df_beijing["pm2.5"]
X = df_beijing.loc[:, df_beijing.columns != "pm.25"]
X = X.join(pd.get_dummies(X["cbwd"])).drop("cbwd", axis="columns")

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=123)

# Average CV score on the training set was: -1.3155717593272511e-27
exported_pipeline = make_pipeline(
    StandardScaler(),
    LassoLarsCV(normalize=False)
)

exported_pipeline.fit(X_train, y_train)
results = exported_pipeline.predict(X_test)
print("Got a final MSE of ", mse(y_test, results))