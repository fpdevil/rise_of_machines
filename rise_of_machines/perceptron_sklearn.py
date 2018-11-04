#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Description
-----------

Perceptron implementation using scikit-learn

Authors: Sampath Singamsetty

:module: perceptron_sklearn
:created: Sun Oct 28 11:31:18 2018
:copyright: Copyright © 2018 Sampath Singamsetty
:license: MIT
:moduleauthor: Sampath Singamsetty

"""
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, mean_squared_error, explained_variance_score

import numpy as np
import matplotlib.pyplot as plt

from helpers import plot_classifier

# load the iris dataset from the sklearn library
iris = datasets.load_iris()

# get details about the features and the target names
print("\n#### Feature Data ####")
print("Feature names from the dataset: {}".format(iris.feature_names))
print("Target names from the dataset: {}".format(iris.target_names))

# assign the petal length and petal width as the feature vectors to X
X = iris.data[:, [2, 3]]
# assign the target vectors to y
y = iris.target

# The   target  class   labels  are   stored  as   ['setosa',  'versicolor',
# 'virginica'] within the dataset. We  will capture the unique integer class
# labels from it.

print("Unique class labels: {}".format(np.unique(y)))

# For the purpose of checking how well the trained model will perform on
# sample  unseen test  data, we  will  split the  dataset into  seperate
# training and testing data sets
# we will split the datasets in 70% - 30% ratio, also the below function
# provides a shuffled data.
# Here X_train will have {no. of samples, no. of features}
#      y_train will have {no. of samples}
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y)

# using  stratification  (stratify=y  above)  we  can  ensure  that  the
# returned split data have the same  proportion of class labels as the input
# dataset for both training and testing datasets.
# the same may be checked by the numpy's bincount method.
print('Count of Labels in y: ', np.bincount(y))
print('Count of Labels in y_train: ', np.bincount(y_train))
print('Count of Labels in y_test: ', np.bincount(y_test))

# As  a part  of optimization  of the  algorithm, we  can apply  feature
# scaling,  by  standardizing  features   using  StandardScaler  class  from
# sklearn's preprocessing module
scaler = StandardScaler()
# first  compute the  mean  and  std for  each  feature  dimension from  the
# training dataset to be used for feature scaling
scaler.fit(X_train)
# standardize by centering and scaling
X_train_std = scaler.transform(X_train)
X_test_std = scaler.transform(X_test)

# With  the standardization  of the  training dataset  complete, we  can
# start the process of training the Perceptron model
classifier = Perceptron(max_iter=40, eta0=0.01, random_state=1)
classifier.fit(X_train_std, y_train)

# having trained the dataset using Perceptron, we can predict for test data
y_predict = classifier.predict(X_test_std)
# with this we can get a list of misclassified samples
errors_ = y_test != y_predict
print('Number of Misclassified samples: %d' % errors_.sum())

# calculate the classification accuracy of the Perceptron on test dataset
# it checks if predicted values for y_test match the actual ones
acs = accuracy_score(y_test, y_predict)
mse = mean_squared_error(y_test, y_predict)
evs = explained_variance_score(y_test, y_predict)
# mean accuracy on given test dataset and test labels
ma = classifier.score(X_test_std, y_test)

print("\n**** Model Performance ****")
print('Accuracy score: %.2f' % acs)
print('Mean Squared Error: %.4f' % mse)
print('Explained Variance Score: %.4f' % evs)
print('Mean Accuracy: %.2f' % ma)

# Now run the perceptron model with the standardized training Data
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

plot_classifier(
    X=X_combined_std,
    y=y_combined,
    classifier=classifier,
    margin=1.0,
    step_size=0.01,
    test_idx=range(105, 150),
    cmap=plt.cm.Paired)

title = ('Perceptron based Training Model using SkLearn')
plt.title(title)
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')

plt.tight_layout()
#  plt.savefig('../images/perceptron_sklearn.png',dpi=300)
plt.show()
