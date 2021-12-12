import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.svm import LinearSVR
from tpot.builtins import StackingEstimator

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=None)

# Average CV score on the training set was: -4.208671414189331
exported_pipeline = make_pipeline(
    PCA(iterated_power=10, svd_solver="randomized"),
    StackingEstimator(estimator=LinearSVR(C=20.0, dual=True, epsilon=0.1, loss="squared_epsilon_insensitive", tol=1e-05)),
    RandomForestRegressor(bootstrap=True, max_features=0.6000000000000001, min_samples_leaf=8, min_samples_split=7, n_estimators=100)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
