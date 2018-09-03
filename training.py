# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 14:19:44 2018

@author: Usman
"""

from sklearn.datasets import load_iris

# import LinearSVC class
from sklearn.svm import LinearSVC

# import KNeighborsClassifier class
from sklearn.neighbors import KNeighborsClassifier

# Load dataset
iris = load_iris()

# assign variables for more convinient handling
X = iris.data
y = iris.target

# create an instance of the LinearSVC classifier
clf = LinearSVC()

# train the model
clf.fit(X, y)

# get the accuracy score of the LinearSVC classifier
print(clf.score(X, y))

# predict the response given new obeservation
print(clf.predict([[6.3, 3.3, 6.0, 2.5]]))

# create an instance of KNeighborClassifier
# The default number of K neighbors is 5
# This can be changed by passing n_neighbors=k as argument
kknDefault = KNeighborsClassifier() # K = 5

# train the model
kknDefault.fit(X, y)

# get the accuracy score of KNeighborsClassifer with K = 5
print(kknDefault.score(X, y))

# predict the response given new observation
print(kknDefault.predict([[6.3, 3.3, 6.0, 2.5]]))

# lets try a different number of neighbors
kknBest = KNeighborsClassifier(n_neighbors=10)

# train the model
kknBest.fit(X, y)

# get the accuracy score of KNeighborsCLassifier with K = 10
print(kknBest.score(X, y))

# predict the response given new observation
print(kknBest.predict([[6.3, 3.3, 6.0, 2.5]]))

# lets try a different number of neighbors
kknWorst = KNeighborsClassifier(n_neighbors=100)

# train the model
kknWorst.fit(X, y)

# get the accuracy score of KNeighborsClassifier with K = 100
print(kknWorst.score(X, y))

# predict the response given new observation
print(kknWorst.predict([[6.3, 3.3, 6.0, 2.5]]))
