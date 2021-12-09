# Auto ML

This project implements auto ml based on simulated annealing.

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