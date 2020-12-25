#!/usr/bin/python
# -*- coding: utf-8 -*-

# importing all the neccesarry dependencies

import os
import sys

import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
import pickle

# loading the model, data, and feature_columns

train = pd.read_csv('extra/formatted_train.csv')
submission = pd.read_csv('extra/formatted_submission.csv')

model = pickle.load(open('extra/model.sav', 'rb'))

feature_columns = pd.read_csv('extra/feature_columns.csv').values.tolist()
for i in range(len(feature_columns)):
    feature_columns[i] = feature_columns[i][0]

# making the predictions for the training data it was trained on

train['Predictions'] = model.predict(train[feature_columns])

# calculating and evaluating the loss of the model wrt the training data

mse = mean_squared_error(train['FVC'], train['Predictions'], squared=False)
mae = mean_absolute_error(train['FVC'], train['Predictions'])

print('MSE Loss:', mse)
print('MAE Loss:', mae)

# using the laplace log metric of the model's performance wrt its training data

def competition_metric(trueFVC, predFVC, predSTD):
    clipSTD = np.clip(predSTD, 70, 9e9)
    deltaFVC = np.clip(np.abs(trueFVC - predFVC), 0, 1000)
    return np.mean(-1 * (np.sqrt(2) * deltaFVC / clipSTD)
                   - np.log(np.sqrt(2) * clipSTD))


print ('Competition metric: ', competition_metric(train['FVC'].values,
       train['Predictions'], mse))

# predicting the actual testing data

submission['FVC'] = model.predict(submission[feature_columns])

# formatting the dataframe to only contain the necessary columns

submission = submission[['Patient_Week', 'FVC']]
submission['Confidence'] = mse

# writing the result to file

submission.to_csv('submission.csv', index=False)
