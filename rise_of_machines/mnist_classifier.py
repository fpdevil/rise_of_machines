#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Description
-----------

Training and interpreting the MNIST dataset from https://www.openml.org/d/554
for predicting the handwritten digits

Authors: Sampath Singamsetty

:module: mnist_decision_tree
:created: Sun Feb 24 11:52:39 2019
:copyright: Copyright © 2019 Sampath Singamsetty
:license: MIT
:moduleauthor: Sampath Singamsetty

"""
import time

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from helpers import plot_classifier

from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.datasets import base, fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

print(__doc__)

# Turn down for faster convergence
t0 = time.time()

# static variable declation
train_size = 50000
test_size = 10000
data_loc = '../data'

# set a seed for the computer's pseudorandom number generator, which would
# allow us to reproduce the results from our script
np.random.seed(123)

# fetch data from https://www.openml.org/d/554
mnist = fetch_openml(
    'mnist_784', version=1, data_home=data_loc, return_X_y=False)
print('fetch the MNIST data to: {}'.format(base.get_data_home()))

# using only a fraction of the dataset
# X = mnist.data[::30]
# y = mnist.target[::30]
X = mnist.data
y = mnist.target

X = X / 255.0  # min and max for X 0 to 255
y = y.astype('int32')

print('MNIST Dataset details:\n {}'.format(mnist.details))

# shuffling of the dataset
# random_state = check_random_state(0)
# permutation = random_state.permutation(X.shape[0])
# X = X[permutation]
# y = y[permutation]

target_names = np.unique(y)
print('Image data shape {}'.format(X.shape))
print('Label data shape {}'.format(y.shape))
print('Target values: {}'.format(target_names))

# X = X.reshape((X.shape[0], -1))
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    train_size=train_size,
    test_size=test_size,
    random_state=42,
    shuffle=True)

# Convert 1-dimensional class arrays to 10-dimensional class matrices
#Y_train = np_utils.to_categorical(y_train, 10)
#Y_test = np_utils.to_categorical(y_test, 10)

# data preprocessing scale the data
scaler = StandardScaler()
# compute the mean and std to be used for later scaling
# fit only to the training data
scaler.fit(X_train)
# perform standardization by centering and scaling
# Now apply the transformations to the data:
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))

clf = LogisticRegression(
    solver='sag', C=1e5, multi_class='multinomial', penalty='l2', tol=0.1)

clf.fit(X_train, y_train)
# plot_classifier(X=X_combined, y=y_combined, classifier=clf)

# coef = Coefficient of the features in the decision function.
sparsity = np.mean(clf.coef_ == 0) * 100

# using the model to make predictions with the test data
y_predicted = clf.predict(X_test)
print("Logistic Regression")
# print(classification_report(y_test, y_predicted, target_names=target_names))
# print("Classification report for classifier %s:\n%s\n" %
#       (classification_report(y_test, y_predicted, target_names=target_names)))

score = clf.score(X_test, y_test)

print("Sparsity with L2 penalty: %.2f%%" % sparsity)
print("Mean accuracy score on the given test data and labels: %.4f" % score)

coef = clf.coef_.copy()
plt.figure(figsize=(10, 5))
scale = np.abs(coef).max()
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(
        coef[i].reshape(28, 28),
        interpolation='nearest',
        cmap=plt.cm.Spectral,
        vmin=-scale,
        vmax=scale)
    plt.title('class %i' % i)
    plt.xticks(())
    plt.yticks(())


def show_images(images, labels):
    cols = min(5, len(images))
    rows = len(images) // cols
    sfig = plt.figure(figsize=(8, 8))
    for z in range(rows * cols):
        sp = sfig.add_subplot(rows, cols, z + 1)
        plt.axis('off')
        plt.imshow(images[z])
        sp.set_title(labels[z])


r = np.random.permutation(len(X))
r = r[:20]
show_images(X[r].reshape(-1, 28, 28), y[r])

fig, ax = plt.subplots(10, 10, figsize=(10, 10))
for i in range(10):
    for j in range(10):
        ax[i][j].imshow(
            X[np.random.randint(X.shape[0])].reshape(28, 28),
            aspect='auto',
            cmap=plt.cm.Spectral)
        ax[i][j].axis('off')

plt.title('Training the MNIST...')

running_time = time.time() - t0
print('Time taken to run classifier: %.3f s' % running_time)

index = 0
misclassifications = []
for label, predict in zip(y_test, y_predicted):
    if label != predict:
        misclassifications.append(index)
    index += 1
print("Number of Misclassified samples: {}".format(len(misclassifications)))

accuracy = metrics.accuracy_score(y_test, y_predicted)
print('Accuracy of the model: {:.2f}'.format(accuracy))

plt.figure(figsize=(20, 4))
for pltIndex, wrongIndex in enumerate(misclassifications[0:10]):
    plt.subplot(2, 5, pltIndex + 1)
    plt.imshow(np.reshape(X_test[wrongIndex], (28, 28)), cmap=plt.cm.autumn)
    plt.title(
        'Predicted: {}, \nActual: {}'.format(y_predicted[wrongIndex],
                                             y_test[wrongIndex]),
        fontsize=14)

# confusion matrix
cm = metrics.confusion_matrix(y_test, y_predicted)

plt.figure(figsize=(9, 9))
sns.heatmap(cm, annot=True, fmt="0.3f", linewidths=0.5, square=True)
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
titles = 'Accuracy Score: {0}'.format(score)
plt.title(titles, size=15)
plt.show()
