"""
Description
-----------

Classification with Logistic Regression using Gradient Descent

running:
python3 logisticregression_with_gd.py --trainer "sklearn"
python3 logisticregression_with_gd.py --trainer "default"

Authors: Sampath Singamsetty

:module:logisticregression_with_gd.py
:created: Thu Nov 08 23:34:46 IST 2018
:copyright: Copyright © 2018 Sampath Singamsetty
:license: MIT
:moduleauthor:Sampath Singamsetty <Singamsetty.Sampath@gmail.com>
"""
import argparse

import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn import model_selection
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from helpers import plot_classifier

ap = argparse.ArgumentParser()
ap.add_argument(
    "-t",
    "--trainer",
    required=True,
    help="pass trainer as either default or sklearn")
args = vars(ap.parse_args())


class LogisticRegressionGD():
    """Logistic Regression classifier using Gradient Descent.

    Paramaters
    -----------
    eta: float
    Learning rate (between 0.0 and 1.0)
    A default value of 0.05 is used if none specified.

    epochs: int
    Number passes or iterations done over  the training data set a default
    value of 100 is used as default if none specified

    random_state : int
    Random number generator seed for initializing the random weights

    Attributes
    -----------
    w_: 1 dimensional array
    Weights after fitting the data.
    cost_ : list
    contains the logistic cost function value during each epochs
    calculated as the sum of squares

    """

    def __init__(self, eta=0.05, epochs=100, random_state=1):
        super(LogisticRegressionGD, self).__init__()
        self.eta = eta
        self.epochs = epochs
        self.random_state = random_state

    def train(self, X, y):
        """Train the classification model and fit the training data
        The data is processed using a sigmoid based activation function

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
        # draw random samples from normal gaussian distribution and
        # initialize the weights with those random values
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        # create an empty place holder for holding the cost or error values
        # through which we may know whether algorithm converges after training
        self.cost_ = []

        # loop through the number of epochs or passes
        for _ in range(self.epochs):
            # calculate the weighted sum of input
            net_input = self.net_input(X)
            # here the activation function is the sigmoid
            output = self.activation(net_input)
            # error is calculated as under
            # error = (true value - activation_function(output))
            errors = (y - output)
            # Now update the Weights
            # the gradient descent is calculated based on the whole training
            # dataset via self.eta * errors.sum() for the bias unit (w_0 = 0)
            # and via self.eta + X.T.dot(errors) for the weights 1 to m where
            # X.T.dot(errors)  is a matrix-vector  multiplication between the
            # feature matrix and eror vector
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()

            # calculate the logistic cost which is the output of log-likelihood
            cost = -(y.dot(np.log(output)) + (1 - y).dot(np.log(1 - output)))
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        """Calculate net input, which is the weighted sum of
        the input and weight vectors added to the bias
        """
        bias = self.w_[0] * 1
        return np.dot(X, self.w_[1:]) + bias

    @classmethod
    def activation(cls, z):
        """Calculation of the Logistic Sigmoid Activation
        """
        # limit the values of z between -250 to 250
        return 1.0 / (1.0 + np.exp(-np.clip(z, -250, 250)))

    def predict(self, X):
        """Return the class label after the unit step

        Essentially, we  make predictions  using activation  function (sigmoid
        function) for the given data X. The function checks the output against
        a Threshold or the condition of w.x + b is checked against 0.0

        prediction = 1 if activation >= 0.0 else 0 (A binary)
        """
        return np.where(self.net_input(X) >= 0.0, 1, 0)


# get the input arg value from --trainer
trainer = args["trainer"]

# application of the training mode over iris dataset
iris = datasets.load_iris()

# list the feature names
print("--------------------------------------------------")
print("   LOGISTIC REGRESSION CLASSIFICATION for IRIS    ")
print("--------------------------------------------------")
print("iris feature names: {}".format(list(iris.feature_names)))
# list the target names
print("iris target names: {}".format(list(iris.target_names)))

X = iris.data[:, [2, 3]]
y = iris.target
print('unique target class labels of iris dataset:', np.unique(y))

# split the data between training and test
X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y)

print('after splitting the data into training and testing')
print('Count of labels in y:', np.bincount(y))
print('Count of labels in y_train:', np.bincount(y_train))
print('Count of labels in y_test:', np.bincount(y_test))

if args["trainer"] == 'sklearn':
    print('\n## using {} implementation of Logistic Regression ##'.format(
        args["trainer"]))

    # data standardization
    scaler = StandardScaler()
    # compute the mean and std to be used for later scaling
    scaler.fit(X_train)
    # perform standardization by centering and scaling
    X_train_std = scaler.transform(X_train)
    X_test_std = scaler.transform(X_test)
    # compute the mean value
    print('Mean of the training data: ', scaler.mean_)
    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))

    # create an instance of the classifier and fit the data
    logreg = linear_model.LogisticRegression(
        solver='liblinear', C=100, random_state=1)
    # the value C which is regularization is a way to prevent overiftting
    logreg.fit(X_train_std, y_train)

    plot_classifier(
        X=X_combined_std,
        y=y_combined,
        classifier=logreg,
        test_idx=range(105, 150),
        cmap=plt.cm.autumn)
    title = 'Logistic Regression using Gradient Descent (Scikit Learn)'

    pp = logreg.predict_proba(X_test_std[:3, :])
    print(
        '\n** predicted probabilities of each class for first 4 samples: {}*'.
        format(pp))
    print("predicted class labels of the above samples: {}".format(
        pp.argmax(axis=1)))
else:
    print('\n## using the {} LogisticRegressionGD ##'.format(args["trainer"]))
    print('## to use scikit-learn model, supply args --trainer sklearn ##')
    # Because  logistic regression  works only  for the  binary classification
    # tasks,  let's consider  only Iris-setosa  and Iris-versicolor  flowers
    # (classes 0 and 1) and check the implementation
    X_train_subset_01 = X_train[(y_train == 0) | (y_train == 1)]
    y_train_subset_01 = y_train[(y_train == 0) | (y_train == 1)]

    # create an instance of the classifier
    logreg = LogisticRegressionGD(eta=0.05, epochs=1000, random_state=1)
    # train and fit the data
    logreg.train(X_train_subset_01, y_train_subset_01)

    plot_classifier(
        X=X_train_subset_01,
        y=y_train_subset_01,
        classifier=logreg,
        cmap=plt.cm.Spectral)
    title = 'Logistic Regression using Gradient Descent (Default)'

plt.title(title)
plt.xlabel(iris.feature_names[2])
plt.ylabel(iris.feature_names[3])

plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
