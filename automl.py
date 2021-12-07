from math import exp
import sklearn.model_selection
import pandas as pd
import sklearn.model_selection
import time
from random import random as rnd
import copy
from multiprocessing import Pool
from ElasticNetManager import ElasticNetManager
from MlpManager import MlpManager
from RidgeManager import RidgeManager
from NestablePool import NestablePool
from sklearn.metrics import mean_squared_error as mse

class AutoML:
    #models to use: Ridge, Neural Network, KNN
    #performance measure to use: MSE
    crossValidation = sklearn.model_selection.KFold(n_splits=5, shuffle=True, random_state=None)
    max_runtime_seconds = None
    start_time = None
    logging_enabled = False
    done = False
    best_model = None

    def __init__(self, logging_enabled = False, max_runtime_seconds = 3600):
        self.logging_enabled = logging_enabled
        self.max_runtime_seconds = max_runtime_seconds

    def fit(self, X, y):
        if not self.do_validation(X, y):
            return
        self.start_time = time.time()
        solution_managers = [
            MlpManager(len(X.columns)),
            RidgeManager(),
            ElasticNetManager(),
        ]
        with NestablePool(5) as pool:
            results = pool.starmap(
                self.simulated_annealing,
                map(lambda manager: [manager, X, y],
                solution_managers))
            print(results)
            
            #selecting the best of the created models
            best_res = None
            for result in results:
                if(best_res == None or result["best_performance"] < best_res["best_performance"]):
                    best_res = result
            
            print("Got the best results with ", best_res["manager"].__class__.__name__)
            print("Parameters used were: ", best_res["best_params"])
            self.best_model = best_res["manager"].fit(best_res["best_params"], X, y)

        self.done = True

    def predict(self, X_new):
        if not self.done:
            print("Please train the model before trying to predict data with it!")
        else:
            return self.best_model.predict(X_new)

    def do_validation(self, X, y):
        for index, value in X.isna().sum().items():
            if(value != 0):
                print("Aborting! The dataset contains missing values in column \""+ str(index) + "\".")
                return False
        if(y.isna().sum() > 0):
            print("Aborting! The target variable contains missing values!")
            return False
        return True

    def simulated_annealing(self, solution_manager, X, y):
        best_performance = None

        current_params = solution_manager.generate_params()

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
            "current_params": current_params,
            "manager": solution_manager
        }       
    
    def crossval(self, manager, params, X, y):
        if(self.logging_enabled):
                print("Starting crossval with parameters ", params)
        starParams = map(lambda indeces: [indeces[0], indeces[1], params, X, y],
                        self.crossValidation.split(X))
        with Pool(5) as pool:
            mses = pool.starmap(manager.fit_and_get_mse, starParams)
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
    automl = AutoML(logging_enabled=True, max_runtime_seconds=60)
    automl.fit(X_train, y_train)
    y_pred = automl.predict(X_test)
    result = mse(y_test, y_pred)
    print("Final model got a MSE of ", result)