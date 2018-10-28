#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Description
-----------

Implementation of large scale machine learning and stochastic gradient descent.

In ADALINE  the cost  function is  minimized by  taking a  step in  the opposite
direction of a cost gradient, calculated  from the whole training set. With very
large data sets spanning across a few million data points, this is not efficient
as the whole  training dataset is reevaluated  each time we take  a steo towards
the  global  minimum.  So,  Stochastic  Gradient Descent  is  introduced  as  an
alternate to the Batch Gradient Descent.

Authors: Sampath Singamsetty

:module: adaptive_sgd_classifier.py
:created: Tue Oct 23 18:24:49 2018
:copyright: Copyright © 2018 Sampath Singamsetty
:license: MIT
:moduleauthor: Sampath Singamsetty

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from helpers import draw_decision_boundary


class AdalineSGD():
    """Adaptive Linear Neuron for Large scale data
    using Stochastic Gradient Descent.

    Parameters
    ---------
    eta: float
    The learning rate (between 0.0 to 1.0)
    a default value of 0.01 is used if none specified

    epochs: int
    Number passes or iterations done over  the training data set a default
    value of 10 is used as default if none specified

    shuffle: bool
    A boolean value which indicates whether to shuffle the training data
    after every apoch to prevent cycles. It has a default of True.
    In order to obtain satisfying results via Stochastic Gradient Descent,
    it is  important to present  it the training  data in a  random order;
    also, we  would like to  shuffle the training  set for every  epoch to
    prevent cycles.

    random_state : int
    Random number generator seed for initializing the random weights

    Attributes
    ----------
    w_: 1 dimensional array
    Weights after fitting the data.
    cost_: list
    SSE - Sum of Squared Errors based cost function value averaged over
    all training samples in every epoch

    """

    def __init__(self, eta=0.01, epochs=10, shuffle=True, random_state=1):
        """Class initialized with the values for
        eta, epochs, shuffle, random_state"""
        self.eta = eta
        self.epochs = epochs
        self.shuffle = shuffle
        self.random_state = random_state
        # whether weights are initialized or not
        self.w_initialized = False

    def train(self, X, y):
        """Train the classification model and fit the training data
        The data is processed using an activation function defined as
        activation = sum(weight_i * x_i) + bias

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
          Training vectors, where n_samples is the number of samples and
                                  n_features is the number of features.
        y : array-like, shape = [n_samples]
          Target values, which are the desired output from algorithm
        Returns
        -------
        self : object
        """
        # Draw random samples from a normal (Gaussian) distribution.
        # initialize the weights with these values
        self._initialize_weights(X.shape[1])
        # initialize an empty place holder for cost values through which
        # we can know whether the algorithm converges after training
        self.cost_ = []
        for _ in range(self.epochs):
            if self.shuffle:
                # shuffle the training data if shuffle=True
                X, y = self._shuffle_data(X, y)
            cost = []
            for x_i, target in zip(X, y):
                cost.append(self._update_weights(x_i, target))
            mean_cost = sum(cost) / len(y)
            self.cost_.append(mean_cost)
        return self

    def partial_train(self, X, y):
        """Train  the  classification  model and  fit  the  training data
        without re-initializing the weights. This can be typically useful
        if  we want  to update our model for an online  learning scenario
        with streaming data.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
          Training vectors, where n_samples is the number of samples and
                                  n_features is the number of features.
        y : array-like, shape = [n_samples]
          Target values, which are the desired output from algorithm
        Returns
        -------
        self : object
        """
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])

        # flatten the array y and check shape
        if y.ravel().shape[0] > 1:
            for x_i, target in zip(X, y):
                self._update_weights(x_i, target)
        else:
            self._update_weights(X, y)
        return self

    def net_input(self, X):
        """Calculate the net input, which is the weighted sum of
        the input and weight vectors added to the bias
        """
        bias = 1 * self.w_[0]
        weighted_sum = np.dot(X, self.w_[1:])
        return weighted_sum + bias

    @classmethod
    def activation(cls, X):
        """Calculation of the Continuous Linear Activation
        :returns: X
        """
        return X

    def predict(self, X):
        """Return the class label after the unit step
        Essentially, we make  predictions using activation function (unit step
        function) for the given data X. The function checks the output against
        a Threshold or the condition of w.x + b is checked against 0

        prediction = 1 if activation >= 0.0 else -1
        """
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)

    def _initialize_weights(self, m):
        """Initialize the input weights to smaller random numbers"""
        # generate a single random value (for random_state=None) based on
        # the chosen probability distribution
        self.rgen = np.random.RandomState(self.random_state)
        # random samples from normal Gaussian distribution
        self.w_ = self.rgen.normal(loc=0.0, scale=0.01, size=1 + m)
        self.w_initialized = True

    def _shuffle_data(self, X, y):
        """Shuffle the training data
        """
        r = self.rgen.permutation(len(y))
        return X[r], y[r]

    def _update_weights(self, x_i, target):
        """Apply the ADALINE learning rule for updating the weights"""
        output = self.activation(self.net_input(x_i))
        # error = (true value - activation function output)
        error = (target - output)
        self.w_[1:] += self.eta * x_i.dot(error)
        self.w_[0] += self.eta * error
        cost = (error**2) / 2.0
        return cost


# Training of the Perceptron model on the IRIS Dataset from UCI repository
# use the dataframe from pandas to load the iris data set
df = pd.read_csv(
    'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
    header=None)
print("iris.data size {}".format(df.shape))
print("iris.data types: {}".format(df.dtypes))
print("----------------------------------")

# Now  extract  the  first  100  class  labels  that  correpond  to  the  50
# Iris-Setosa and 50 Iris-Versicolor flowers,  respectively and convert  the
# class labels into two integer class labels, 1 for versicolor and -1 for setosa.
# These values will be assigned to vector y
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)

# Similarly  extract the  first feature  column (sepal  length) and  third feature
# column (petal  length) of those  100 training samples and  assign the same  to a
# feature matrix X
X = df.iloc[0:100, [0, 2]].values

# standardize the feature vector
# Standardization of  dataset enables  the individual feature  look like
# standard normally  distributed data:  Gaussian with  zero mean  and unit
# variance.
X_std = np.copy(X)
X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

# testing the Implementation
sgd = AdalineSGD(eta=0.01, epochs=15, random_state=1)
clf = sgd.train(X_std, y)

# decision boundary printing
title = ('Decision surface at ADALINE - Stochastic Gradient Descent')
draw_decision_boundary(X_std, y, classifier=clf)
plt.title(title)
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.xticks()
plt.yticks()

plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

# average cost over the number of epochs
plt.plot(range(1, len(sgd.cost_) + 1), sgd.cost_, marker='*')
plt.title('Cost Minimization Rate')
plt.xlabel('Epochs')
plt.ylabel('Average Cost')
plt.xticks()
plt.yticks()

plt.tight_layout()
plt.show()
