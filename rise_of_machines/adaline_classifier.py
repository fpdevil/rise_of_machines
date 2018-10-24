#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

ADALINE Classifier
-----------------

An implementation of the Adaptive Linear Neurons and the convergence of learning
using python. In this kind of learning, the cost function is minimized by taking
a step in  the opposite direction of  a cost gradient calculated  from the whole
training set, which is why it's called as a Batch Gradient Descent.

Authors: Sampath Singamsetty

:module: adaline_classifier
:created:
:platform: OS X
:copyright: Copyright © 2018 Sampath Singamsetty
:license: MIT, see LICENSE for details.
:moduleauthor: Sampath Singamsetty <Singamsetty.Sampath@gmail.com>
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from helpers import create_meshgrid


class Adaline():
    """ADaptive LInear NEuron Classifier.

    Parameters
    ----------
    eta: float
    The learning rate (between 0.0 to 1.0)
    a default value of 0.01 is used if none specified

    epochs: int
    Number passes or iterations done over  the training data set a default
    value of 10 is used if none specified

    random_state : int
    Random number generator seed for initializing the random weights

    Attributes
    ----------
    w_: 1 dimensional array
    Weights after fitting.
    cost_: list
    SSE - Sum of Squared Errors based cost function value for every epoch
    """

    def __init__(self, eta=0.01, epochs=50, random_state=1):
        self.eta = eta
        self.epochs = epochs
        self.random_state = random_state

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
        # a pseudo random number generator
        rgen = np.random.RandomState(self.random_state)
        # Draw random samples from a normal (Gaussian) distribution.
        # initialize the weights with these values
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        # initialize an empty place holder for cost values through which
        # we can know whether the algorithm converges after training
        self.cost_ = []

        # loop through the number of posses
        for _ in range(self.epochs):
            net_input = self.net_input(X)
            # Here the activation function has no effect on the code as
            # it is simply an identify function. It only serves as a place
            # holder as the same function can be different in the case of
            # Logistic Regression. There it can be changed to a Sigmoid
            # function to implement the Logistic Regression Classifier.
            output = self.activation(net_input)
            # error = (true value - linear activation function output)
            errors = (y - output)
            # update the weights
            # the gradient descent is calculated based on the whole training
            # dataset via self.eta * errors.sum() for the bias unit (w_0 = 0)
            # and via self.eta + X.T.dot(errors) for the weights 1 to m where
            # X.T.dot(errors)  is a matrix-vector  multiplication between the
            # feature matrix and error vector
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        """Calculate net input, which is the weighted sum of
        the input and weight vectors added to the bias
        """
        bias = 1 * self.w_[0]
        return np.dot(X, self.w_[1:]) + bias

    def activation(self, X):
        """Calculation of the Continuos Linear Activation
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


# def create_meshgrid(x, y, margin=1, step=0.02):
#     x_min, x_max = x.min() - margin, x.max() + margin
#     y_min, y_max = y.min() - margin, y.max() + margin
#     xx, yy = np.meshgrid(
#     np.arange(x_min, x_max, step), (np.arange(y_min, y_max, step)))
#     return xx, yy


def draw_decision_boundary(X, y, classifier, margin=1, step=0.02):
    # set-up grid for plotting.
    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = create_meshgrid(X0, X1, margin, step)

    mesh = np.array([xx.ravel(), yy.ravel()])
    print("np.array: {}", format(mesh))
    Z = classifier.predict(mesh.T)
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    for cl in np.unique(y):
        print("cl: ", cl)
        plt.scatter(
            x=X[y == cl, 0],
            y=X[y == cl, 1],
            alpha=0.8,
            label=cl,
            edgecolor='black')


# Training of the Perceptron model on the IRIS Dataset from UCI repository
# use the dataframe from pandas to load the iris data set
df = pd.read_csv(
    'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
    header=None)
print("iris.data size {}".format(df.shape))
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

# plotting of cost against the number of epochs for different
# learning paths
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

adaline1 = Adaline(epochs=10, eta=0.01).train(X, y)
adaline2 = Adaline(epochs=10, eta=0.0001).train(X, y)

ax1.plot(
    range(1,
          len(adaline1.cost_) + 1), np.log10(adaline1.cost_), marker='o')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('log(Sum-Squared-Errors)')
ax1.set_title('ADALINE - Learning Rate 0.01')

ax2.plot(range(1, len(adaline2.cost_) + 1), adaline2.cost_, marker='o')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Sum of Squared Errors')
ax2.set_title('ADALINE - Learning Rate 0.0001')

plt.show()

# standardize features
X_std = np.copy(X)
X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

# decision boundary printing
ada = Adaline(epochs=20, eta=0.01)
clf = ada.train(X_std, y)
title = ('Decision surface at Adaline - Gradient Descent ')
draw_decision_boundary(X_std, y, classifier=clf)
plt.title(title)
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.legend(loc='upper left')
plt.xticks()
plt.yticks()

plt.tight_layout()
plt.show()

plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Average Cost')

plt.tight_layout()
plt.show()
