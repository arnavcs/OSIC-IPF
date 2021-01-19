#!/usr/bin/python
# -*- coding: utf-8 -*-

# loading the dependencies

import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf

# loading the model, data, feature_columns, and fvc_scale

train = pd.read_csv('extra/formatted_train.csv')
submission = pd.read_csv('extra/formatted_submission.csv')

model = tf.keras.models.load_model('extra/model.h5')

feature_columns = pd.read_csv('extra/feature_columns.csv'
                              ).values.tolist()
for i in range(len(feature_columns)):
    feature_columns[i] = feature_columns[i][0]

fvc_scale = int(open('extra/fvc_scale.txt', 'r').read())

# getting the mae, mse, and loss of the model on the training data

(loss, mae, mse) = model.evaluate(train[feature_columns], train['FVC'],
                                  verbose=0)

print ('MSE Loss:', mse)
print ('MAE Loss:', mae)

# making predictions on the training data, and rescales the FVC to be back to the normal scale

train['Predictions'] = model.predict(train[feature_columns]) * fvc_scale
train['FVC'] = train['FVC'] * fvc_scale


# calculates the competition metric on the training set

def competition_metric(trueFVC, predFVC, predSTD):
    clipSTD = np.clip(predSTD, 70, 9e9)
    deltaFVC = np.clip(np.abs(trueFVC - predFVC), 0, 1000)
    return np.mean(-1 * (np.sqrt(2) * deltaFVC / clipSTD)
                   - np.log(np.sqrt(2) * clipSTD))


print ('Competition metric: ', competition_metric(train['FVC'].values,
       train['Predictions'], 215))

# predicting the FVC for the testing data

submission['FVC'] = model.predict(submission[feature_columns]) \
    * fvc_scale

# formatting the dataframe to only contain the necessary columns

submission = submission[['Patient_Week', 'FVC']]
submission['Confidence'] = 215

# writing the result to file

submission.to_csv('submission.csv', index=False)
