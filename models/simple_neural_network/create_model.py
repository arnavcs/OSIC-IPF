#!/usr/bin/python
# -*- coding: utf-8 -*-

# importing all the neccesarry libraries

import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf

# reading the important data from files

train = pd.read_csv('extra/formatted_train.csv')

feature_columns = pd.read_csv('extra/feature_columns.csv'
                              ).values.tolist()
for i in range(len(feature_columns)):
    feature_columns[i] = feature_columns[i][0]


# making the basic neural network model

def build_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(12,
            activation=tf.nn.relu,
            input_shape=[len(feature_columns)]),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1),
        ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)
    model.compile(loss='mse', 
                  optimizer=optimizer, 
                  metrics=['mae', 'mse']
                  )

    return model


model = build_model()

# training the model

history = model.fit(train[feature_columns], train['FVC'], epochs=500)

# saving the model to a file

model.save('extra/model.h5')
