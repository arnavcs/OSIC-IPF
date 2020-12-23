# importing all the neccesarry dependencies
import os
import sys

import numpy as np
import pandas as pd

from sklearn import linear_model, ensemble
from sklearn.metrics import mean_squared_error, mean_absolute_error

import plotly.express as px
import chart_studio.plotly as py
import plotly.graph_objs as go
from plotly.offline import iplot
import cufflinks
cufflinks.go_offline()
cufflinks.set_config_file(world_readable=True, theme='pearl')

import tensorflow.compat.v2.feature_column as fc
import tensorflow as tf

# reads the data
train_df = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv')
test_df = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')
submission_df = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/sample_submission.csv')

# remove duplicates from train
train_df.drop_duplicates(keep=False, inplace=True, subset=['Patient','Weeks'])

# adding the Patient and Weeks features to the submission
submission_df['Patient'] = submission_df['Patient_Week'].apply(lambda x:x.split('_')[0])
submission_df['Weeks'] = submission_df['Patient_Week'].apply(lambda x: int(x.split('_')[-1]))

# taken from "Feature engineering with a linear model" by Matt
submission_df =  submission_df[['Patient','Weeks', 'Confidence','Patient_Week']]
submission_df = submission_df.merge(test_df.drop('Weeks', axis=1), on="Patient")

# labels all the datasets
train_df['Dataset'] = 'train'
test_df['Dataset'] = 'test'
submission_df['Dataset'] = 'submission'

# merging the datasets
all_data = train_df.append([test_df, submission_df])

all_data = all_data.reset_index()
all_data = all_data.drop(columns=['index'])

# making the FirstWeek column to represent the week of the first FVC measurement
all_data['FirstWeek'] = all_data['Weeks']
all_data.loc[all_data.Dataset=='submission','FirstWeek'] = np.nan
all_data['FirstWeek'] = all_data.groupby('Patient')['FirstWeek'].transform('min')



# finds the first Percent value of each patient and adds a feature to eac row containing that value
first_percent = all_data.loc[all_data.Weeks == all_data.FirstWeek][['Patient','Percent']].rename({'Percent': 'FirstPercent'}, axis=1).groupby('Patient').first().reset_index()

all_data = all_data.merge(first_percent, on='Patient', how='left')



# a feature to represent the number of weeks passed since the first checkup
all_data['WeeksPassed'] = all_data['Weeks'] - all_data['FirstWeek']



# calculates the approximate height of the patient as a feature to be used by the model
def calculate_height(row):
    if row['Sex'] == 'Male':
        return row['FirstFVC'] / (27.63 - 0.112 * row['Age'])
    else:
        return row['FirstFVC'] / (21.78 - 0.101 * row['Age'])

all_data['Height'] = all_data.apply(calculate_height, axis=1)



# converts the sex and smoking status features into seperate binary features
all_data = pd.concat([
    all_data,
    pd.get_dummies(all_data.Sex),
    pd.get_dummies(all_data.SmokingStatus)
], axis=1)

all_data = all_data.drop(columns=['Sex', 'SmokingStatus'])


# scale the features so that they are between 0 and 1
def scale_feature(series):
    return (series - series.min()) / (series.max() - series.min())

all_data['Weeks'] = scale_feature(all_data['Weeks'])
all_data['Percent'] = scale_feature(all_data['Percent'])
all_data['Age'] = scale_feature(all_data['Age'])
all_data['FirstWeek'] = scale_feature(all_data['FirstWeek'])
all_data['FirstFVC'] = scale_feature(all_data['FirstFVC'])
all_data['WeeksPassed'] = scale_feature(all_data['WeeksPassed'])
all_data['Height'] = scale_feature(all_data['Height'])

# the features that will be used to predict FVC
feature_columns = [
#     'Percent',       including this makes the whole model rely on FirstPrecent and Percent
    'Age',
    'FirstWeek',
    'FirstFVC',
    'FirstPercent',
    'WeeksPassed',
    'Height',
    'Female',
    'Male', 
    'Currently smokes',
    'Ex-smoker',
    'Never smoked',
]



# seperating the data into 3 seperate sections again
train = all_data.loc[all_data.Dataset == 'train']
test = all_data.loc[all_data.Dataset == 'test']
submission = all_data.loc[all_data.Dataset == 'submission']



# making the linear regression model
model = linear_model.HuberRegressor(max_iter=20000)




# making the predictions for the training data it was trained on
predictions = model.predict(train[feature_columns])

# a graph of the coeffcients of the model's features

px.bar(x=train[feature_columns].columns.values, y=model.coef_, labels={'x':'Features', 'y':'Model Coefficient'}).show()

# calculating and evaluating the loss of the model wrt the training data
mse = mean_squared_error(
    train['FVC'],
    predictions,
    squared=False
)

mae = mean_absolute_error(
    train['FVC'],
    predictions
)

print('MSE Loss: {0:.2f}'.format(mse))
print('MAE Loss: {0:.2f}'.format(mae))



# using the laplace log metric of the model's performance wrt its training data
def competition_metric(trueFVC, predFVC, predSTD):
    clipSTD = np.clip(predSTD, 70 , 9e9)  
    deltaFVC = np.clip(np.abs(trueFVC - predFVC), 0 , 1000)  
    return np.mean(-1 * (np.sqrt(2) * deltaFVC / clipSTD) - np.log(np.sqrt(2) * clipSTD))
    

print('Competition metric: ', competition_metric(train['FVC'].values, predictions, mse))



# adding the predictions of the training set to the training set
train['prediction'] = predictions

# graphing the predictions vs the real FVC

px.scatter(train, x='prediction', y='FVC').show()

# graphing the error of the model 

delta = predictions - train['FVC']
delta.iplot(kind='hist', xTitle='Error (FVC)', yTitle='Frequency')



# predicting the actual testing data
sub_predictions = model.predict(submission[feature_columns])
submission['FVC'] = sub_predictions




# formatting the dataframe to only contain the necessary columns
submission = submission[['Patient_Week', 'FVC']]
submission['Confidence'] = mse

submission.head()



# writing the result to file
submission.to_csv('submission.csv', index=False)

