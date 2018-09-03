# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 12:40:01 2018

@author: Usman
"""

from sklearn.datasets import load_iris
iris = load_iris()

# The feature (column) names and response
print(iris.feature_names)
print(iris.target)
print(iris.target_names)

# The object types of the feature matrix and response array
print(type(iris.data))
print(type(iris.target))

# The shapes of the sample features
print(iris.data.shape)
print(iris.target.shape)