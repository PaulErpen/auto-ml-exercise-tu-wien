# Auto ML

This project implements Auto-ML based on simulated annealing.
There are also scripts that utilize auto-sklearn and tpot on the same data sets.
This enables us to compare the performance of our own Auto-Ml solution to others.

## Structure of this Repository

The ``AutoML`` directory contains code for our own Auto-ML solution.

The ``CSVLogsProcessing`` directory contains a jupyter notebook to Process the CSV logs.

The ``data`` directory contains all the data sets. The fifa data set is included in its preprocessed state.

The ``DumpedModels`` directory contains the results of the Auto-ML training.
Sadly we weren't able to include the trained models for autosklearn and our Auto-ML. This were dumps created with joblib and had very huge sizes. A single dump could be up to 250 MB.

The ``eval`` directory contains scripts to retrieve the performance from the dumped model.

The ``output`` directory contains the CSV-logs created by our own AutoML solution.

The ``Preprocessing`` directory contains the Jupyter Notebook used to preprocess the FIFA data set.

The ``training`` directory contains all the scripts for training the Auto-ML models.

## How to use

The code for the Auto-ML solution is located in the ``AutoML`` directory.
There are example scripts of how to use all of the three Auto-ML solutions in the training directory.
For the beijing2.5 data for example, the scripts are located in the ``training/beijing`` directory. These scripts also contain comments to explain whats goin on.

Each script does the following:
* Load the dataset
* Minimally preprocess the data set so it only contains numerical data and no missing values
* Train the Auto-ML solution
* Compute the MSE for the model computed by the Auto-ML solution
* Save the model to a file that can be loaded again

In order to analyze the custom Auto-ML solution it also logs its parameter tuning process in multiple CSV-files.
How the CSV-files are structured is described in the ```Tuning outputs``` section.
In order for each data set to have its own CSV-logs, please provide the custom Auto-ML solution with a unique directory to save the files to.
If this isn't done the new CSV-logs will override the ones generated for another data set.
Please make sure that the directory exists, before starting the Auto-ML solution.

## Tuning outputs

The auto ml process takes a long time and generates lots of data.
For monitoring purposes the parameter tuning process can be dumped to a csv file.
Each of the machine learning solution will dump its parameters to its own file.

The schema for the different solution are as follows:
* Multilayer Perceptron
```
timestamp,temperature,rmse,n_neurons_l1,n_neurons_l2,alpha,is_best,is_current
```
* ElasticNet
```
timestamp,temperature,rmse,alpha,l1,is_best,is_current
```
* Ridge
```
timestamp,temperature,rmse,alpha,is_best,is_current
```