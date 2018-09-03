# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 14:42:14 2018

@author: Usman
"""

import pandas as pd

from sklearn.datasets import load_iris
iris = load_iris()

# readfile and attribute list into variables
data = pd.DataFrame(data=iris['data'], columns=iris['feature_names'])

print(data.describe())
