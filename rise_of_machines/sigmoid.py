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
plt.grid()

plt.rc('axes', labelsize=13)
plt.rc('font', size=15)

plt.title('Logistic Sigmoid')
plt.xlabel('z')
plt.ylabel('g(z)')
plt.text(-7.5, 0.85, r'$g(z) = \frac{1}{1 + e^{-z}}$')

plt.axvline(linewidth=2, color='g', linestyle='dashed')
plt.axhline(y=0.5, color='r', linestyle='dashed')

plt.xticks()
plt.yticks(np.arange(0.0, 1.0, 0.1))
plt.grid(True)

plt.show()
