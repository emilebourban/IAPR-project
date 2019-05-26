import numpy as np
import matplotlib.pyplot as plt
import skimage
import pickle
# import tensorflow as tf
# import keras

from keras.models import load_model

import Utilities
from model import *

im_train, *_ = Utilities.load_images_annotations()

model = load_model('unet_varroas.hdf5')
train_set = np.expand_dims(np.array(Utilities.resize_images(im_train[:10], (256, 256), 1)), -1)
pred = model.predict(train_set)

plt.figure(figsize=(10, 10))

for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(pred[i][...,0], cmap='gray')

plt.show()























