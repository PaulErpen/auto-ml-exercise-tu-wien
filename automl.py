from math import exp
import sklearn.model_selection
import pandas as pd
import sklearn.model_selection
import time
from random import random as rnd
import copy
from multiprocessing import Process
from multiprocessing import Queue
from multiprocessing import Pool
from MlpManager import MlpManager

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
        res_mlp = self.simulated_annealing("mlp", X, y)
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

    def simulated_annealing(self, type, X, y):
        best_performance = None

        solution_managers = {
            "mlp": MlpManager(len(X.columns))
        }

        solution_manager = solution_managers[type]
        current_params = solution_manager.generate_params(X)

        #These are the base stats
        best_performance = self.crossval(solution_manager, current_params, X, y)
        best_params = copy.deepcopy(current_params)
        #This is the tracker for the solution that might not be the best, but is definetly the one that originated from the parameters
        current_performance = best_performance

        temp = self.current_temperature()
        while(temp > 0):
            if(self.logging_enabled):
                print("Temperature = ", temp)
            #Taking a step in a random "direction"
            new_params = solution_manager.parameter_step(current_params, temp)
            
            candidate_performance = self.crossval(solution_manager, new_params, X, y)

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
    
    def crossval(self, manager, params, X, y):
        if(self.logging_enabled):
                print("Starting crossval with parameters ", params)
        starParams = map(lambda indeces: [indeces[0], indeces[1], params, X, y],
                        self.crossValidation.split(X))
        with Pool() as pool:
            mses = pool.starmap(manager.fit, starParams)
            mean_mse = sum(mses) / len(mses)
            if(self.logging_enabled):
                print("Crossval results ", mses, " with mean ",mean_mse )       
            return mean_mse

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
