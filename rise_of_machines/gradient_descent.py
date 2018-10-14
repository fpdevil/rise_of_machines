#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (C) 2018

# Author       : Sampath Singamsetty
# Created Time : Sat Oct  6 22:06:47 2018
# File Name    : gradient_descent.py
# Description  : Using Gradient descent in Linear regression
########################################################################

#  An example of  modelling a line commanded by  y = mx +c through a  set of points
#  with m = slope and c= y-intercept.The goal  is to find the best values for m and
#  c. A  standard approach to solving  this type of  problem is to define  an error
#  function (also called a cost function) that measures how “good” a given line is.
#  This function will take  in a (m,c) pair and return an error  value based on how
#  well the  line fits  our data.  To compute this  error for  a given  line, we’ll
#  iterate through each  (x,y) point in our  data set and sum  the square distances
#  between each point’s y value and the  candidate line’s y value (computed at mx +
#  c). It’s conventional to square this distance  to ensure that it is positive and
#  to make our error function differentiable.

def get_error(intercept, slope, points):
    """get_error function computes the error for a line passing through a
    given set of points (x, y)

    :intercept: y-intercept of the line passing through a set of points
    :slope: slope of the equation represented as m
    :points: set of (x, y) coordinates
    :returns: the error value computed

    """
    error_value = 0
    for i in range(0, len(points)):
        error_value += (points[i].y - (slope * points[i].x + intercept)) ** 2
    return error_value / float(len(points))
