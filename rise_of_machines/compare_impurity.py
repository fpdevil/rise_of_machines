#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Description
-----------

Comparision of the Impurity criteria based on splitting

Authors: Sampath Singamsetty

:module:decision_tree.py
:created: Sat Feb 16 23:58:45 CST 2019
:copyright: Copyright © 2019 Sampath Singamsetty
:license: MIT
:moduleauthor:Sampath Singamsetty <Singamsetty.Sampath@gmail.com>
"""
import numpy as np
import matplotlib.pyplot as plt


def gini(p):
    """Calculate the Gini impurity
    :param p: The probability or proportion of the samples belonging to a class
    :returns: Gini Index
    :rtype: float
    """
    return p * (1 - p) + (1 - p) * (1 - (1 - p))


def entropy(p):
    """Calculate the Entropy for a sample
    :param p: The probability or proportion of the samples belonging to a class
    :returns: The Entropy
    :rtype: float
    """
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def error(p):
    """Calculate the classification error which is a measure of impurity
    :param p: The probability or proportion of the samples belonging to a class
    :returns: Classification error
    :rtype: float
    """
    return 1 - np.max([p, 1 - p])


# list with step of 0.01 from 0 to 1.0
x = np.arange(0.0, 1.0, 0.01)

# calculate the impurity measures based on the defined
# functions for a set of values
ent = [entropy(p) if p != 0 else None for p in x]
sc_ent = [e * 0.5 if e else None for e in ent]
err = [error(i) for i in x]
gini_idx = gini(x)

idx = [ent, sc_ent, gini_idx, err]
labels = [
    'Entropy', 'Entropy (scaled)', 'Gini Impurity', 'Misclassification Error'
]
symbols = ['-', ':', '--', '-.']
colorl = ['blue', 'green', 'red', 'cyan', 'pink']

fig = plt.figure()
ax = plt.subplot(111)
for i, lab, s, c in zip(idx, labels, symbols, colorl):
    line = ax.plot(x, i, label=lab, linestyle=s, lw=2, color=c)

ax.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, 1.15),
    ncol=5,
    fancybox=True,
    shadow=False)
ax.axhline(y=0.5, linewidth=1, color='y', linestyle='--')
ax.axhline(y=1.0, linewidth=1, color='y', linestyle='--')

ax.annotate(
    'Entropy',
    xy=(0.19, 0.7),
    xycoords='data',
    xytext=(0.3, 0.7),
    arrowprops=dict(arrowstyle="simple", fc="0.6", ec="none"),
    textcoords='data')

ax.annotate(
    'Entropy (scaled)',
    xy=(0.2, 0.36),
    xycoords='data',
    xytext=(0.2, 0.55),
    arrowprops=dict(arrowstyle="simple", fc="0.6", ec="none"),
    textcoords='data')

ax.annotate(
    'Gini Impurity',
    xy=(0.82, 0.29),
    xycoords='data',
    xytext=(0.7, 0.55),
    arrowprops=dict(arrowstyle="simple", fc="0.6", ec="none"),
    textcoords='data')

ax.annotate(
    'Misclassification Error',
    xy=(0.7, 0.3),
    xycoords='data',
    xytext=(0.4, 0.2),
    arrowprops=dict(arrowstyle="simple", fc="0.6", ec="none"),
    textcoords='data')

plt.ylim([0, 1.1])
plt.xlabel('p[i=1]')
plt.ylabel('Impurity Index')
plt.show()
