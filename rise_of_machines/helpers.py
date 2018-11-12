#!/usr/bin/env python3
# coding: utf-8

# Contains common methods frequently used across....
# The example reference at the below matplotlib is helpful in choosing an
# appropriate colormap for the output plot
# https://matplotlib.org/examples/color/colormaps_reference.html

# import the necessary packages
import numpy as np
import matplotlib.pyplot as plt


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
    step_size: float This would be the buffer for clarity
    This is spacing between values. For any output out, this is the distance
    between two adjacent values, out[i+1] - out[i]
    alpha: float
    color alpha value
    cmap: color map
    """
    # set-up the marker generator and color map for plotting
    markers = ('s', 'o', 'x', '^', 'v')

    # for data, first set-up a grid for plotting.
    X0, X1 = x[:, 0], x[:, 1]
    xx, yy = create_meshgrid(X0, X1, margin, step)

    mesh = np.array([xx.ravel(), yy.ravel()])
    print("np.array: {}", format(mesh))
    # compute the classifiers output
    Z = classifier.predict(mesh.T)
    Z = Z.reshape(xx.shape)
    # now plot the contour
    plt.contourf(xx, yy, Z, alpha=alpha, cmap=cmap)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    for idx, cl in enumerate(np.unique(y)):
        print("cl: ", cl)
        plt.scatter(
            x=x[y == cl, 0],
            y=x[y == cl, 1],
            alpha=alpha,
            marker=markers[idx],
            label=cl,
            edgecolor='yellow')


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
    # set-up the marker generator for plotting
    markers = ('s', 'o', 'x', '*', 'v')

    # setup and define a range for plotting the data
    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = create_meshgrid(X0, X1, margin=margin, step=step_size)

    # compute the output of the classifier
    mesh = np.c_[xx.ravel(), yy.ravel()]
    mesh_output = classifier.predict(mesh)

    # reshape the array
    mesh_output = mesh_output.reshape(xx.shape)

    # draw and fill contour lines
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

    # plotting and highlighting the test samples
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
