#!/usr/bin/python
# -*- coding: utf-8 -*-

# importing all the neccesarry dependencies

import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import linear_model
import pickle

# reading the data and the feature_columns

train = pd.read_csv('formatted_train.csv')

feature_columns = pd.read_csv('feature_columns.csv').values.tolist()
for i in range(len(feature_columns)):
    feature_columns[i] = feature_columns[i][0]

# making the linear regression model

model = linear_model.HuberRegressor(max_iter=20000)

# training the model

model.fit(train[feature_columns], train['FVC'])

# saving the model

pickle.dump(model, open('model.sav', 'wb'))
