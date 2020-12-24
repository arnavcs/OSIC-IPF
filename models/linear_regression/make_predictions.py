#!/usr/bin/python
# -*- coding: utf-8 -*-
# importing all the neccesarry dependencies

import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
import pickle

# import plotly.express as px
# import chart_studio.plotly as py
# import plotly.graph_objs as go
# from plotly.offline import iplot
# import cufflinks
# cufflinks.go_offline()
# cufflinks.set_config_file(world_readable=True, theme='pearl')

# loading the model, data, and feature_columns

train = pd.read_csv('formatted_train.csv')
submission = pd.read_csv('formatted_submission.csv')

model = pickle.load(open('model.sav', 'rb'))

feature_columns = pd.read_csv('feature_columns.csv').values.tolist()
for i in range(len(feature_columns)):
    feature_columns[i] = feature_columns[i][0]

# making the predictions for the training data it was trained on

predictions = model.predict(train[feature_columns])

# a graph of the coeffcients of the model's features
# px.bar(x=train[feature_columns].columns.values, y=model.coef_, labels={'x':'Features', 'y':'Model Coefficient'}).show()

# calculating and evaluating the loss of the model wrt the training data

mse = mean_squared_error(train['FVC'], predictions)

mae = mean_absolute_error(train['FVC'], predictions)

print('MSE Loss:', mse)
print('MAE Loss:', mae)


# using the laplace log metric of the model's performance wrt its training data

def competition_metric(trueFVC, predFVC, predSTD):
    clipSTD = np.clip(predSTD, 70, 9e9)
    deltaFVC = np.clip(np.abs(trueFVC - predFVC), 0, 1000)
    return np.mean(-1 * (np.sqrt(2) * deltaFVC / clipSTD)
                   - np.log(np.sqrt(2) * clipSTD))


print ('Competition metric: ', competition_metric(train['FVC'].values,
       predictions, mse))

# adding the predictions of the training set to the training set

train['prediction'] = predictions

# graphing the predictions vs the real FVC
# px.scatter(train, x='prediction', y='FVC').show()

# graphing the error of the model
# delta = predictions - train['FVC']
# delta.iplot(kind='hist', xTitle='Error (FVC)', yTitle='Frequency')

# predicting the actual testing data

sub_predictions = model.predict(submission[feature_columns])
submission['FVC'] = sub_predictions

# formatting the dataframe to only contain the necessary columns

submission = submission[['Patient_Week', 'FVC']]
submission['Confidence'] = mse

# writing the result to file

submission.to_csv('submission.csv', index=False)
