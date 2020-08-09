#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Description
-----------
Decision Tree classifier for the Iris flower dataset

Authors: Sampath Singamsetty

:module:iris_decisiontree_classifier.py
:created: Thu Feb 21 08:23:46 CST 2019
:copyright: Copyright © 2019 Sampath Singamsetty
:license:
:moduleauthor:Sampath Singamsetty <Singamsetty.Sampath@gmail.com>
"""
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets, metrics, model_selection
from sklearn.tree import DecisionTreeClassifier, export_graphviz

# for visualization
from pydotplus import graph_from_dot_data

# load the Iris flower dataset
iris = datasets.load_iris()

# distribute between class and target
X = iris.data[:, [2, 3]]
y = iris.target

# get class label information of the flower dataset
print("Iris class labels: {}".format(np.unique(y)))
# print list of feature names
print("Iris feature names: {}".format(list(iris.feature_names)))
# print list of target names
print("Iris target names: {}".format(list(iris.target_names)))

# split the daaset into training an test datasets
X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y)

# print additional details
print('\n# after splitting the data into training and testing #')
print(' count of labels in y:', np.bincount(y))
print(' count of labels in y_train:', np.bincount(y_train))
print(' count of labels in y_test:', np.bincount(y_test))


# Some helper functions for plotting the data
def create_meshgrid(x, y, margin=1, step=0.02):
    """Create a numoy rectangular meshgrid out of an array of
    x values and an array of y values

    @ref https://stackoverflow.com/questions/36013063
                 /what-is-the-purpose-of-meshgrid-in-python-numpy

    :x: array-like point x
    :y: array-like point y
    :margin: (int) boundary
    :step: (float) stepping the values, default = 0.02

    Examples
    --------
    x = np.array([0, 1, 2, 3, 4])
    y = np.array([0, 1, 2, 3, 4])
    xx,yy=np.meshgrid(x,y)
    plt.plot(xx,yy, marker='.', color='k',linestyle='none')

    """
    x_min, x_max = x.min() - margin, x.max() + margin
    y_min, y_max = y.min() - margin, y.max() + margin
    # define the mesh grid, with xx and yy holding the grid of
    # points where the function will be evaluated
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))
    return xx, yy


# A helper for plotting the decision surface
def plot_classifier(X,
                    y,
                    classifier,
                    margin=1.0,
                    step_size=0.01,
                    alpha=0.8,
                    test_idx=None,
                    cmap=plt.cm.Paired):
    """Draw the datapoints and boundaries
    Parameters
    ----------
    x: {array-like}, shape = [n_samples, n_features]
    y: array-like, shape = [n_samples]
    margin: margin for the min and max
    step_size: float
    This is spacing between values. For any output out, this is the distance
    between two adjacent values, out[i+1] - out[i]
    alpha: float
    blending value to decide transparency - 0 (transparent) and 1 (opaque)
    test_idx: list
    cmap: object
    color map for the output colors of objects
    """
    # setup marker generator for plotting
    markers = ('s', 'o', 'x', '*', 'v')

    # setup and define a range for plotting the data
    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = create_meshgrid(X0, X1, margin=margin, step=step_size)

    # compute the output of the classifier
    mesh = np.c_[xx.ravel(), yy.ravel()]
    mesh_output = classifier.predict(mesh)

    # reshape the array
    mesh_output = mesh_output.reshape(xx.shape)

    # draw and fill the contour lines
    plt.contourf(xx, yy, mesh_output, alpha=0.4, cmap=cmap)

    # now overlay the training coordinates over the plot
    # set boundaries
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks((np.arange(int(min(X[:, 0]) - 1), int(max(X[:, 0]) + 1), 1.0)))
    plt.yticks((np.arange(int(min(X[:, 1]) - 1), int(max(X[:, 1]) + 1), 1.0)))

    # use a separate marker for each training label
    for (i, cl) in enumerate(np.unique(y)):
        plt.scatter(
            x=X[y == cl, 0],
            y=X[y == cl, 1],
            alpha=alpha,
            marker=markers[i],
            label=cl,
            edgecolors='purple')

    # for plotting and highlighting the test samples
    if test_idx:
        # x_test, y_test = X[test_idx, :], y[test_idx]
        x_test = X[test_idx, :]
        plt.scatter(
            x_test[:, 0],
            x_test[:, 1],
            c='',
            edgecolors='purple',
            alpha=alpha,
            linewidths=1,
            marker='o',
            s=100,
            label='Test Data')


# Decision Tree with the training data
tree = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=1)
tree.fit(X_train, y_train)

print('cross validation score: {}'.format(
    model_selection.cross_val_score(tree, X, y, cv=10)))

X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))
plot_classifier(
    X=X_combined, y=y_combined, classifier=tree, test_idx=range(105, 150))

# Testing Accuracy
y_predicted_test = tree.predict(X_test)

# To check ho accurate was our classifier on the testing set
# as because of the variation with each run, it may give varied
# results with testing
output = metrics.accuracy_score(y_test, y_predicted_test)
print("\nDecision Tree Classifier - Testing Accuracy:", round(output, 4))

title = 'Decision Tree Classifier'
plt.title(title)
plt.xlabel(iris.feature_names[2])
plt.ylabel(iris.feature_names[3])
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

# visualize the decision tree in the form of a png file
dot_data = export_graphviz(
    tree,
    filled=True,
    rounded=True,
    special_characters=True,
    class_names=iris.target_names,
    feature_names=['petal length', 'petal width'],
    out_file=None)
graph = graph_from_dot_data(dot_data)
graph.write_png('../images/iris_decision_tree.png')
