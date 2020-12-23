# importing all the neccesarry dependencies
import os
import sys

import numpy as np
import pandas as pd

# reads the data
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
submission_df = pd.read_csv('../input/sample_submission.csv')

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
