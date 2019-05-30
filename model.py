import os

import numpy as np
import skimage.io as io
import skimage.transform as trans
from keras import backend as keras
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.layers import *
from keras.models import *
from keras.optimizers import *


def ConvNet(input_size, pretrained_weights=None, summary=False):
    """Simple convolutional network followed by fully connected layers similar to LeNet arctitecture
        - input_size: (x_size, y_size, n_channels)
        - pretrained weights if loading from a file"""

    inputs = Input(input_size)

    conv1 = Conv2D(8, 3, activation='tanh', padding='valid', kernel_initializer = 'he_normal')(inputs)
    pool1 = MaxPool2D(pool_size=(2, 2), padding='valid')(conv1)

    conv2 = Conv2D(16, 5, activation='tanh', padding='valid', kernel_initializer = 'he_normal')(pool1)
    pool2 = MaxPool2D(pool_size=(2, 2), padding='valid')(conv2)

    conv3 = Conv2D(32, 5, activation='tanh', padding='valid', kernel_initializer = 'he_normal')(pool2)
    pool3 = MaxPool2D(pool_size=(2,2), padding='valid')(conv3)

    flat = Flatten()(pool2)

    dense1 = Dense(300, activation='tanh', kernel_initializer = 'he_normal')(flat)
    drop1 = Dropout(0.4)(dense1)
    dense2 = Dense(50, activation='tanh', kernel_initializer = 'he_normal')(drop1)
    drop2 = Dropout(0.2)(dense2)
    dense3 = Dense(2, activation='softmax', kernel_initializer = 'he_normal')(drop2)

    model = Model(inputs=inputs, outputs=dense3)
    model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

    if summary:
        model.summary()

    if pretrained_weights is not None:
        model.load_weights(pretrained_weights)

    return model


    
# def ConvNet(input_size, pretrained_weights=None, summary=False):

#     inputs = Input(input_size)

#     conv1 = Conv2D(8, 5, activation='tanh', padding='valid', kernel_initializer = 'he_normal')(inputs)
#     pool1 = MaxPool2D(pool_size=(2, 2), padding='valid')(conv1)

#     conv2 = Conv2D(16, 5, activation='tanh', padding='valid', kernel_initializer = 'he_normal')(pool1)
#     pool2 = MaxPool2D(pool_size=(3, 3), padding='valid')(conv2)

#     conv3 = Conv2D(32, 5, activation='tanh', padding='valid', kernel_initializer = 'he_normal')(pool2)
#     pool3 = MaxPool2D(pool_size=(3, 3), padding='valid')(conv3)

#     flat = Flatten()(pool2)

#     dense1 = Dense(300, activation='tanh', kernel_initializer = 'he_normal')(flat)
#     drop1 = Dropout(0.4)(dense1)
#     dense2 = Dense(50, activation='tanh', kernel_initializer = 'he_normal')(drop1)
#     drop2 = Dropout(0.2)(dense2)
#     dense3 = Dense(2, activation='softmax', kernel_initializer = 'he_normal')(drop2)

#     model = Model(inputs=inputs, outputs=dense3)
#     model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

#     if summary:
#         model.summary()

#     if pretrained_weights is not None:
#         model.load_weights(pretrained_weights)

#     return model
