#!/usr/bin/python
# -*- coding: utf-8 -*-

# importing all the neccesarry libraries

import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
import pydicom

import cv2
from tqdm.notebook import tqdm

# reads the data and the encoder model

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
submission = pd.read_csv('../input/sample_submission.csv')

encoder = tf.keras.models.load_model('encoder.h5')

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

first_percent = all_data.loc[all_data.Weeks == all_data.FirstWeek][
        ['Patient',
        'Percent']
        ].rename(
                {'Percent': 'FirstPercent'},
                axis=1
        ).groupby('Patient').first().reset_index()
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

# removes all rows which are of patients with bad ids

all_data = all_data[all_data['Patient'] != 'ID00011637202177653955184']
all_data = all_data[all_data['Patient'] != 'ID00052637202186188008618']

# uses the encoder to create more features

unique_patients = all_data.loc[all_data.Weeks == all_data.FirstWeek]

def add_encoder_features(row):
    x = []
    dataset_field = row.Dataset
    if dataset_field == 'submission':
        dataset_field = 'test'
    for i in os.listdir(f'../input/osic-pulmonary-fibrosis-progression/{dataset_field}/{row.Patient}'):
        d = pydicom.dcmread(f'../input/osic-pulmonary-fibrosis-progression/{dataset_field}/{row.Patient}/{i}')
        x.append(cv2.resize(d.pixel_array / 2**10, (512, 512)))
    x = np.expand_dims(x, axis=-1)
    r = encoder.predict(x)
    ind = 0
    for j in range(30):
        for k in range(8):
            for l in range(8):
                ind = 8 * 8 * j + 8 * k + l
                if (len(r) < 30):
                    row[f'f{ind}'] = 0
                elif (len(r[j]) < 8):
                    row[f'f{ind}'] = 0
                elif (len(r[j]) < 8):
                    row[f'f{ind}'] = 0
                else:
                    row[f'f{ind}'] = r[j][k][l][0]
    return row

unique_patients = unique_patients.apply(add_encoder_features, axis=1)

up_features = ['Patient']
for i in range(30 * 8 * 8):
    up_features.append(f'f{i}')
unique_patients = unique_patients[up_features]

all_data = all_data.merge(unique_patients, on='Patient', how='left')

# the features that will be used to predict FVC

feature_columns = [  
#     'Percent',       including this makes the whole model rely on FirstPercent and Percent
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

for i in range(30 * 8 * 8):
    feature_columns.append(f'f{i}')

# saving the feature_columns

pd.DataFrame(feature_columns).to_csv('extra/feature_columns.csv',
        index=False)


# normalizes features

def norm(series):
    return (series - series.mean()) / series.std()

all_data[feature_columns] = all_data[feature_columns].apply(norm)

# scales all the FVC values to be between 0 and 1

fvc_scale = all_data['FVC'].max()
all_data['FVC'] /= fvc_scale

# stores FVC_scale amount in file

open('extra/fvc_scale.txt', 'w').write(str(fvc_scale))

# seperating the data into 3 seperate sections again

train = all_data.loc[all_data.Dataset == 'train']
test = all_data.loc[all_data.Dataset == 'test']
submission = all_data.loc[all_data.Dataset == 'submission']

# writes the data to a file

train.to_csv('extra/formatted_train.csv', index=False)
test.to_csv('extra/formatted_test.csv', index=False)
submission.to_csv('extra/formatted_submission.csv', index=False)
