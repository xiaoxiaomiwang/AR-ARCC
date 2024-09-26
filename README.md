# AR-ARCC: Probabilistic Intervals Prediction based on Adaptive Regression with Attention Residual Connections and Covariance Constraints

## Description

This paper introduces a novel probability interval prediction method called Adaptive Regression with Attention Residual Connection and Covariance Constraint (AR-ARCC). By integrating Monte Carlo and Bayesian methods, we leverage the strengths of both to achieve a more flexible and accurate method for generating prediction intervals. Additionally, through the optimization of the loss function, introduction of penalty terms, and improvement of mean squared error calculations, the model's performance in interval prediction tasks is enhanced. Finally, the integration of an interactive channel heterogeneous self-attention module, combined with residual blocks, enhances the modeling capability of the neural network. The comprehensive application of these methods results in superior performance of the model in handling uncertainty and local variations.
## Usage

This repository contains the following scripts:

* `PIGenerator.py`: Contains the PIGenerator class that is used to perform cross-validation using different NN-based PI-generation methods.        
* `utils.py`: Additional methods used to transform the data and calculate the metrics. 
* `models/NNmodel.py`: Implements the PI-generation methods tested in this work: AR_ARCC.
* `models/network.py`: Defines the network architecture.
* `Demo.ipynb`: Jupyter notebook demo using a synthetic dataset.

## Acknowledgement

We greatly appreciate the GitHub repository of the article `Dual Accuracy-Quality-Driven Neural Network for Prediction Interval Generation` for providing valuable code support.