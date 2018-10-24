#!/usr/bin/env python3
# coding: utf-8

# import the necessary packages
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def create_meshgrid(x, y, margin=1, step=0.02):
    """Create a numoy rectangular meshgrid out of an array of
    x values and an array of y values

    @ref https://stackoverflow.com/questions/36013063/what-is-the-purpose-of-meshgrid-in-python-numpy

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
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))
    return xx, yy


def draw_decision_boundary(x,
                           y,
                           classifier,
                           margin=1,
                           step=0.02,
                           alpha=0.8,
                           cmap=plt.cm.coolwarm):
    """Draw decision boundary separating the collections

    Parameters
    ----------
    x: {array-like}, shape = [n_samples, n_features]
    y: array-like, shape = [n_samples]
    margin: margin for the min and max
    step: float
    This is spacing between values. For any output out, this is the distance
    between two adjacent values, out[i+1] - out[i]
    alpha: float
    color alpha value
    cmap: color map
    """
    # set-up the marker generator and color map for plotting
    markers = ('s', 'o', 'x', '*', 'v')

    # for data, first set-up a grid for plotting.
    X0, X1 = x[:, 0], x[:, 1]
    xx, yy = create_meshgrid(X0, X1, margin, step)

    mesh = np.array([xx.ravel(), yy.ravel()])
    print("np.array: {}", format(mesh))
    Z = classifier.predict(mesh.T)
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.8, cmap=cmap)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    for idx, cl in enumerate(np.unique(y)):
        print("cl: ", cl)
        plt.scatter(
            x=x[y == cl, 0],
            y=x[y == cl, 1],
            alpha=0.8,
            marker=markers[idx],
            label=cl,
            edgecolor='yellow')
