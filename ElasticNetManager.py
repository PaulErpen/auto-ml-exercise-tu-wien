from sklearn.linear_model import ElasticNet
import helpers
from random import random as rnd
import copy
from sklearn.metrics import mean_squared_error as mse

class ElasticNetManager:
    alpha_min, alpha_max = 0, 1
    l1_min, l1_max = 0, 1

    def generate_params(self):
        #the tuning paramters are being saved in a map, so we can deep copy them later and mutate them wihout affecting the base ones
        return {
            "alpha": helpers.get_random_in_range(self.alpha_min, self.alpha_max),
            "l1": helpers.get_random_in_range(self.l1_min, self.l1_max)
        }
    
    def parameter_step(self, current_params, temp):
        new_params = copy.deepcopy(current_params)
        new_params["alpha"] = helpers.clamp(
            new_params["alpha"] + helpers.get_random_in_range(-1, 1) * temp,
            self.alpha_min,
            self.alpha_max)
        new_params["l1"] = helpers.clamp(
            new_params["l1"] + helpers.get_random_in_range(-1, 1) * temp,
            self.l1_min,
            self.l1_max
        )
        return new_params
    
    @staticmethod
    def model_from_params(params):
        return ElasticNet(alpha=params["alpha"], l1_ratio=params["l1"])

    def fit(self, params, X, y):
        model = ElasticNetManager.model_from_params(params)
        model.fit(X, y)
        return model
    
    def fit_and_get_mse(self, train_indeces, test_indeces, params, X, y):
        X_train, X_test = X.iloc[train_indeces], X.iloc[test_indeces]
        y_train, y_test = y.iloc[train_indeces], y.iloc[test_indeces]
        model = self.fit(params, X_train, y_train)
        return mse(y_test, model.predict(X_test))