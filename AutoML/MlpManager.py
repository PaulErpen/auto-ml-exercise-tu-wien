import helpers
from random import random as rnd
import copy
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error as mse

class MlpManager:
    #tuning params ranges
    #n layers and n neurons are somewhat related , i would only optimize for unrelated tuning params
    n_neurons_min, n_neurons_max = None, None
    alpha_min, alpha_max = 0, 1

    def __init__(self, n_columns):
        self.n_neurons_min, self.n_neurons_max = int(n_columns / 2), n_columns * 2

    def generate_params(self):
        #the tuning paramters are being saved in a map, so we can deep copy them later and mutate them wihout affecting the base ones
        return {
            "n_neurons_l_1": helpers.get_random_in_range(self.n_neurons_min, self.n_neurons_max, is_int=True),
            "n_neurons_l_2": helpers.get_random_in_range(self.n_neurons_min, self.n_neurons_max, is_int=True),
            "alpha": helpers.get_random_in_range(self.alpha_min, self.alpha_max)
        }
    
    def parameter_step(self, current_params, temp):
        new_params = copy.deepcopy(current_params)
        new_params["n_neurons_l_1"] = int(
            helpers.get_value_step_with_unsticky(
                new_params["n_neurons_l_1"],
                self.n_neurons_min,
                self.n_neurons_max,
                temp))
        new_params["n_neurons_l_2"] = int(
            helpers.get_value_step_with_unsticky(
                new_params["n_neurons_l_2"],
                self.n_neurons_min,
                self.n_neurons_max,
                temp))
        new_params["alpha"] = helpers.get_value_step_with_unsticky(
                new_params["alpha"],
                self.alpha_min,
                self.alpha_max,
                temp)
        return new_params
    
    @staticmethod
    def model_from_params(params):
        return MLPRegressor(
                        hidden_layer_sizes=[params["n_neurons_l_1"], params["n_neurons_l_2"]], 
                        solver="adam", 
                        alpha=params["alpha"])

    def fit_and_get_mse(self, train_indeces, test_indeces, params, X, y):
        X_train, X_test = X.iloc[train_indeces], X.iloc[test_indeces]
        y_train, y_test = y.iloc[train_indeces], y.iloc[test_indeces]
        model = self.fit(params, X_train, y_train)
        return mse(y_test, model.predict(X_test))

    def fit(self, params, X, y):
        model = MlpManager.model_from_params(params)
        model.fit(X, y)
        return model
