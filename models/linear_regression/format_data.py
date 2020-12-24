#!/usr/bin/python
# -*- coding: utf-8 -*-

# importing all the neccesarry libraries

import os
import sys

import numpy as np
import pandas as pd

# reads the data

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
submission = pd.read_csv('../input/sample_submission.csv')

# adding the Patient and Weeks features to the submission

submission['Patient'] = submission['Patient_Week'].apply(lambda x: \
        x.split('_')[0])
submission['Weeks'] = submission['Patient_Week'].apply(lambda x: \
        int(x.split('_')[-1]))

# adds the features of test onto submission

submission = submission[['Patient', 'Weeks', 'Confidence',
                        'Patient_Week']]
submission = submission.merge(test.drop('Weeks', axis=1), on='Patient')

# labels all the datasets

train['Dataset'] = 'train'
test['Dataset'] = 'test'
submission['Dataset'] = 'submission'

# merging the datasets

all_data = train.append([test, submission])

all_data = all_data.reset_index()
all_data = all_data.drop(columns=['index'])

# making the FirstWeek column to represent the week of the first FVC measurement

all_data['FirstWeek'] = all_data['Weeks']
all_data.loc[all_data.Dataset == 'submission', 'FirstWeek'] = np.nan
all_data['FirstWeek'] = all_data.groupby('Patient')['FirstWeek'
        ].transform('min')

# finds the first FVC measurement of each patient and adds a feature to each row containing that value

first_fvc = all_data.loc[all_data.Weeks
                         == all_data.FirstWeek][['Patient', 'FVC'
        ]].rename({'FVC': 'FirstFVC'}, axis=1).groupby('Patient'
        ).first().reset_index()
all_data = all_data.merge(first_fvc, on='Patient', how='left')

# finds the first Percent value of each patient and adds a feature to each row containing that value

first_percent = all_data.loc[all_data.Weeks
                             == all_data.FirstWeek][['Patient',
        'Percent']].rename({'Percent': 'FirstPercent'},
                           axis=1).groupby('Patient'
        ).first().reset_index()
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

all_data = pd.concat([all_data, pd.get_dummies(all_data.Sex),
                     pd.get_dummies(all_data.SmokingStatus)], axis=1)

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

# seperating the data into 3 seperate sections again

train = all_data.loc[all_data.Dataset == 'train']
test = all_data.loc[all_data.Dataset == 'test']
submission = all_data.loc[all_data.Dataset == 'submission']

# writes the data to a file

train.to_csv('formatted_train.csv', index=False)
test.to_csv('formatted_test.csv', index=False)
submission.to_csv('formatted_submission.csv', index=False)

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

# saving the feature_columns

pd.DataFrame(feature_columns).to_csv('feature_columns.csv', index=False)
