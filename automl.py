from math import exp
import sklearn.model_selection
import pandas as pd
from sklearn.neural_network import MLPRegressor
import sklearn.model_selection
import time
from random import random as rnd
from sklearn.metrics import mean_squared_error as mse
import copy
from multiprocessing import Process
from multiprocessing import Queue
from multiprocessing import Pool

class AutoML:
    #models to use: Ridge, Neural Network, KNN
    #performance measure to use: MSE
    crossValidation = sklearn.model_selection.KFold(n_splits=5, shuffle=True, random_state=None)
    max_runtime_seconds = 30
    start_time = None
    logging_enabled = False

    def __init__(self, logging_enabled = False):
        self.logging_enabled = logging_enabled

    def fit(self, X, y):
        if not self.do_validation(X, y):
            return
        self.start_time = time.time()
        res_mlp = self.train_neural_network(X, y)
        print(res_mlp["best_performance"])
        pass

    def predict(self, X_new):
        pass

    def do_validation(self, X, y):
        for index, value in X.isna().sum().items():
            if(value != 0):
                print("Aborting! The dataset contains missing values in column \""+ str(index) + "\".")
                return False
        if(y.isna().sum() > 0):
            print("Aborting! The target variable contains missing values!")
            return False
        return True

    def train_neural_network(self, X, y):
        best_performance = None

        #tuning params ranges
        #n layers and n neurons are somewhat related , i would only optimize for unrelated tuning params
        n_neurons_min, n_neurons_max = int(len(X.columns) / 2), len(X.columns) * 2
        alpha_min, alpha_max = 0, 1

        #the tuning paramters are being saved in a map, so we can deep copy them later and mutate them wihout affecting the base ones
        current_params = {
            "n_neurons_l_1": self.get_random_in_range(n_neurons_min, n_neurons_max, is_int=True),
            "n_neurons_l_2": self.get_random_in_range(n_neurons_min, n_neurons_max, is_int=True),
            "alpha": self.get_random_in_range(alpha_min, alpha_max)
        }

        #These are the base stats
        best_performance = self.crossval(current_params, X, y)
        best_params = copy.deepcopy(current_params)
        #This is the tracker for the solution that might not be the best, but is definetly the one that originated from the parameters
        current_performance = best_performance

        temp = self.current_temperature()
        while(temp > 0):
            if(self.logging_enabled):
                print("Temperature = ", temp)
            new_params = self.parameter_step_mlp(current_params, temp, n_neurons_min, n_neurons_max)
            
            candidate_performance = self.crossval(new_params, X, y)

            #lower performance is better -> MSE
            #determine if this is the best model
            if(candidate_performance < best_performance):
                best_performance = candidate_performance
                best_params = copy.deepcopy(current_params)
                if(self.logging_enabled):
                    print("New best ", best_performance, " achieved with ", best_params)
            
            #The special step concerning simulated annealing
            performance_difference = candidate_performance - current_performance
            metropolis = exp(-performance_difference / temp)
            if(performance_difference < 0 or rnd() < metropolis):
                current_params = new_params
                current_performance = candidate_performance
                if(self.logging_enabled):
                    print("New current ", current_performance, " achieved with ", current_params)

            temp = self.current_temperature()
        return {
            "best_params": best_params,
            "best_performance": best_performance,
            "current_performance": current_performance,
            "current_params": current_params
        }
    
    def model_from_params(self, params):
        return MLPRegressor(
                        hidden_layer_sizes=[params["n_neurons_l_1"], params["n_neurons_l_2"]], 
                        solver="adam", 
                        alpha=params["alpha"])

    def parameter_step_mlp(self, current_params, temp, n_neurons_min, n_neurons_max):
        tune_which = rnd()
        new_params = copy.deepcopy(current_params)
        if(tune_which < 1/3):
            new_params["n_neurons_l_1"] = int(self.clamp(
                new_params["n_neurons_l_1"] + rnd() * temp * n_neurons_max,
                n_neurons_min,
                n_neurons_max))
        elif(tune_which < 2/3):
            new_params["n_neurons_l_1"] = int(self.clamp(
                new_params["n_neurons_l_1"] + rnd() * temp * n_neurons_max,
                n_neurons_min,
                n_neurons_max))
        else:
            new_params["alpha"] = new_params["alpha"] + rnd() * temp
        return new_params
    
    def clamp(self, val, min_val, max_val):
        return max(min_val, min(max_val, val))
    
    def crossval(self, params, X, y):
        if(self.logging_enabled):
                print("Starting crossval with parameters ", params)
        starParams = map(lambda indeces: [indeces[0], indeces[1], params, X, y],
                        self.crossValidation.split(X))
        with Pool() as pool:
            mses = pool.starmap(self.fit_mlp, starParams)
            mean_mse = sum(mses) / len(mses)
            if(self.logging_enabled):
                print("Crossval results ", mses, " with mean ",mean_mse )       
            return mean_mse

    def fit_mlp(self, train_indeces, test_indeces, params, X, y):
        X_train, X_test = X.iloc[train_indeces], X.iloc[test_indeces]
        y_train, y_test = y.iloc[train_indeces], y.iloc[test_indeces]
        model = self.model_from_params(params)
        model.fit(X_train, y_train)
        return mse(y_test, model.predict(X_test))

    def get_random_in_range(self, min, max, is_int = False):
        diff = max - min
        val = rnd() * diff + min
        if(is_int):
            return int(val)
        else:
            return val

    def elapsed_time(self):
        return time.time() - self.start_time
    
    def current_temperature(self):
        return 1 - self.elapsed_time() / self.max_runtime_seconds


if __name__ == "__main__":
    df_beijing = pd.read_csv("./data/beijing.csv")

    df_beijing = df_beijing.drop("No", axis="columns").dropna()
    y = df_beijing["pm2.5"]
    X = df_beijing.loc[:, df_beijing.columns != "pm.25"]
    X = X.join(pd.get_dummies(X["cbwd"])).drop("cbwd", axis="columns").drop("pm2.5", axis="columns")

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state=None)
    automl = AutoML(logging_enabled=True)
    automl.fit(X_train, y_train)
    #mlp = MLPRegressor()
    #mlp.fit(X=X_train, y=y_train)
