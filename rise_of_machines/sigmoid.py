#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Description
-----------

Logistic Sigmoid function

Authors: Sampath Singamsetty

:module: sigmoid
:created: Tue Nov  6 13:13:16 2018
:copyright: Copyright © 2018 Sampath Singamsetty
:license: MIT
:moduleauthor: Sampath Singamsetty

"""
import math

import numpy as np
import matplotlib.pyplot as plt


def sigmoid(z):
    arr = []
    for i in z:
        arr.append(1.0 / (1.0 + math.exp(-i)))
    return arr


# plotting the sigmoid
x = np.arange(-10., 10., 0.1)
s = sigmoid(x)

plt.plot(x, s)
plt.grid(True)

plt.rc('axes', labelsize=14)
plt.rc('font', size=14)

plt.title("Logistic Sigmoid Activation function", fontsize=14)
plt.xlabel('z')
plt.ylabel(r'$\phi$(z)')

# set some math info on the graph
plt.text(-9.0, 0.85, r'$\phi(z) = \frac{1}{1 + e^{-z}}$')

# text and annotation
props = dict(facecolor='black', shrink=0.1)
plt.text(-9.0, 0.70, r'$z \rightarrow +\infty \Leftrightarrow \phi(z) \rightarrow 1$')
plt.text(-9.0, 0.60, r'$z \rightarrow -\infty \Leftrightarrow \phi(z) \rightarrow 0$')

plt.annotate('Saturating', xytext=(4.5, 0.8), xy=(10, 1),
             arrowprops=props, fontsize=14, ha="center")
plt.annotate('Saturating', xytext=(-4.5, 0.2), xy=(-10, 0),
             arrowprops=props, fontsize=14, ha="center")
plt.annotate('Linear', xytext=(3, 0.3), xy=(0, 0.5),
             arrowprops=props, fontsize=14, ha="center")

# draw dotted lines crossing at (0, 0.5)
plt.axvline(linewidth=2, color='g', linestyle='dashed')
plt.axhline(y=0.5, color='r', linestyle='dashed')

plt.xticks()
plt.yticks(np.arange(0.0, 1.1, 0.1))
plt.grid(True)

# plt.savefig('../images/sigmoid.png')
plt.show()
