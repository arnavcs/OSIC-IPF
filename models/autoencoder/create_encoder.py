#!/usr/bin/python
# -*- coding: utf-8 -*-

# importing the libraries

import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import pydicom
from tensorflow.keras.utils import Sequence
from tensorflow.keras.layers import Dense, Dropout, Activation, \
    Flatten, Input, BatchNormalization, UpSampling2D, Add, Conv2D, \
    MaxPooling2D, LeakyReLU

from tensorflow.keras import Model
from tensorflow.keras.optimizers import Nadam
import cv2
from tqdm.notebook import tqdm

# the training data

train = pd.read_csv('../input/train.csv')
train_data = {}
for p in train.Patient.values:
    train_data[p] = os.listdir(f'../input/train/{p}/')


# the image generator that makes the training data for the encoder

class IGenerator(Sequence):

    BAD_ID = ['ID00011637202177653955184', 'ID00052637202186188008618']

    def __init__(
        self,
        keys=list(train_data.keys()),
        train_data=train_data,
        batch_size=32,
        ):
        self.keys = [k for k in keys if k not in self.BAD_ID]
        self.train_data = train_data
        self.batch_size = batch_size

    def __len__(self):
        return 1000

    def __getitem__(self, idx):
        x = []
        keys = np.random.choice(self.keys, size=self.batch_size)
        for k in keys:
            try:
                i = np.random.choice(self.train_data[k], size=1)[0]
                d = pydicom.dcmread(f'../input/train/{k}/{i}')
                x.append(cv2.resize(d.pixel_array / 2 ** 10, (512,
                         512)))
            except:
                print (k, i)
        x = np.array(x)
        x = np.expand_dims(x, axis=-1)
        return (x, x)


# the function to make the encoder

def get_encoder(shape=(512, 512, 1)):

    def res_block(x, n_features):
        _x = x
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        x = Conv2D(n_features, kernel_size=(3, 3), strides=(1, 1),
                   padding='same')(x)
        x = Add()([_x, x])
        return x

    inp = Input(shape=shape)

    # 512

    x = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same'
               )(inp)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same'
               )(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    # 256

    x = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same'
               )(x)
    for _ in range(2):
        x = res_block(x, 32)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    # 128

    x = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same'
               )(x)
    for _ in range(2):
        x = res_block(x, 32)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    # 64

    x = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same'
               )(x)
    for _ in range(3):
        x = res_block(x, 64)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    # 32

    x = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same'
               )(x)
    for _ in range(3):
        x = res_block(x, 64)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    # 16

    x = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same'
               )(x)
    for _ in range(3):
        x = res_block(x, 128)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    # 8

    x = Conv2D(1, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
    return Model(inp, x)


# the function to make the decoder

def get_decoder(shape=(8, 8, 1)):
    inp = Input(shape=shape)

    # 8

    x = UpSampling2D((2, 2))(inp)
    x = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same'
               )(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    # 16

    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    # 32

    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    # 64

    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    # 128

    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    # 256

    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Conv2D(1, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
    return Model(inp, x)


# instantiates one encoder and one decoder

encoder = get_encoder((512, 512, 1))
decoder = get_decoder((8, 8, 1))

# makes the model for training the encoder

inp = Input((512, 512, 1))
e = encoder(inp)
d = decoder(e)
model = Model(inp, d)

# compiles the model

model.compile(optimizer=Nadam(lr=2 * 1e-3, schedule_decay=1e-5),
              loss='mse')

# fits the model

model.fit_generator(IGenerator(), steps_per_epoch=500, epochs=5)

# saves the encoder

encoder.save('extra/encoder.h5')
