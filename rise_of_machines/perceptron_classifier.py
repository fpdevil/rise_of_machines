#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Perceptron Classifier
-----------------

An implementation of the Perceptron based mode using python
in a Single Layer Neural Network. Iris data set with 2 variables
is used as an example.

Authors: Sampath Singamsetty

:module: perceptron_classifier
:created:
:platform: OS X
:copyright: Copyright © 2018 Sampath Singamsetty
:license: MIT, see LICENSE for details.
:moduleauthor: Sampath Singamsetty <Singamsetty.Sampath@gmail.com>
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap


class Perceptron():
    """Perceptron classifier.

    Parameters
    ----------
    eta: float
    The learning rate (between 0.0 to 1.0)
    a default value of 0.01 is used if none specified

    epoch: int
    number passes or iterations done over  the training data set a default
    value of 10 is used if none specified

    Attributes
    ----------
    w_: 1 dimensional array
    Weights after fitting.
    errors_: list
    number of mis-classifications during each epoch
    """

    def __init__(self, eta=0.01, epoch=10):
        super(Perceptron, self).__init__()
        self.eta = eta
        self.epoch = epoch

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
        # initialize an array with weights equal to zero. The array length
        # is  equal to  the number  of features plus one.  This additional
        # value is the threshold. w_0 * x_0 = 1
        # the 1 can also be considered as bias
        self.w_ = np.zeros(1 + X.shape[1])

        # collect the number of mis-classifications during each epoch
        self.errors_ = []

        # loop for the number of cycles/iterations
        for _ in range(self.epoch):
            errors = 0
            # loop over each training sample x_i and it's target
            # calculate the output value and update the weights
            for x_i, target in zip(X, y):
                # Update the weight and bias at each step. The value for updating
                # the weights at each increment is calculated by the learning rule
                # weight update = learning_rate * (true - predicted)
                # more specifically
                # Δwj = η(target(i) − output(i))x(i)j
                # Δwj = η(target(i) − output(i))xj(i)
                update = self.eta * (target - self.predict(x_i))
                # update the weight wj:=wj+Δwj
                self.w_[1:] += update * x_i
                # update the bias
                self.w_[0] += update
                # aggregate the errors (return true=1 or false=0) until
                # the update nears 0.0
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """Calculate net input, which is the weighted sum of
        the input and weight vectors added to the bias
        """
        bias = 1 * self.w_[0]
        return np.dot(X, self.w_[1:]) + bias

    def predict(self, X):
        """Return the class label after the unit step
        Essentially, we make  predictions using activation function (unit step
        function) for the given data X. The function checks the output against
        a Threshold or the condition of w.x + b is checked against 0

        prediction = 1 if activation >= 0.0 else -1
        """
        return np.where(self.net_input(X) >= 0.0, 1, -1)


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

# data plotting
plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
plt.scatter(
    X[50:100, 0], X[50:100, 1], color='green', marker='x', label='versicolor')

plt.title('Setosa vs Versicolor')
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')

plt.show()

# Perceptron Training
per = Perceptron(eta=0.1, epoch=20)
per.train(X, y)

# plot the Perceptron
print('Total number of misclassifications: {}'.format(len(per.errors_)))
plt.plot(range(1, len(per.errors_) + 1), per.errors_, marker='*')
plt.title('Error Convergence')
plt.xlabel('Epochs')
plt.ylabel('Number of Updates done')
plt.show()

# Plotting the Decision boundaries
#
# We will use the np.meshgrid() functionality from the numpy library
# The problem that you face with arrays is that you need 2-D arrays of x
# and y coordinate  values. With np.meshgrid() function, you  can create a
# rectangular grid out of  an array of x values and an  array of y values:
# the  np.meshgrid() function  takes two  1D  arrays and  produces two  2D
# matrices corresponding to all pairs of  (x, y) in the two arrays. Then,
# you can use these matrices to make all sorts of plots.
#
# x = np.array([0, 1, 2, 3, 4])
# y = np.array([0, 1, 2, 3, 4])
# xx,yy=np.meshgrid(x,y)
# plt.plot(xx,yy, marker='.', color='k',linestyle='none')


def plot_decision_regions(X, y, classifier, margin=1, step=0.02):
    """A helper function to visualize the decision boundaries
    for 2 dimensional data sets

    Parameters
    ----------
    X : {array-like}, shape = [n_samples, n_features]

    y : array-like, shape = [n_samples]

    classifier : Here the Perceptron

    step : float
    This is spacing between values. For any output out, this is the distance
    between two adjacent values, out[i+1] - out[i]

    :returns: object

    """
    # setup the color map and markers
    colors = ('red', 'green', 'purple', 'blue', 'gray')
    markers = ('s', 'x', 'o', '^', 'v')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # now plot the decision surface
    # first extract the 2 dimensional info by determining the
    # minimum and maximum  values for the 2 features  and use
    # those  feature vectors to create a  pair of grid arrays
    # xx1 and xx2.
    x1_min, x1_max = X[:, 0].min() - margin, X[:, 0].max() + margin
    x2_min, x2_max = X[:, 1].min() - margin, X[:, 1].max() + margin

    xx1, xx2 = np.meshgrid(
        np.arange(x1_min, x1_max, step), (np.arange(x2_min, x2_max, step)))

    # flatten the grid arrays xx1 and xx2 using ravel() function
    # inorder to get it processed by the classfier.
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)

    plt.contourf(xx1, xx2, Z, alpha=0.1, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # not plot the class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(
            x=X[y == cl, 0],
            y=X[y == cl, 1],
            alpha=0.8,
            c=colors[idx],
            label=cl,
            edgecolor='black')


# draw the decision boundary
plot_decision_regions(X, y, classifier=per)
plt.title('Decision Boundary')
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')

plt.show()
