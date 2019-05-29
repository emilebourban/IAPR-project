import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import skimage
import pickle
# import tensorflow as tf
# import keras

from keras.models import load_model

import Utilities
from model import *

im_train, im_test, im_val, anno_train, anno_test, anno_val = Utilities.load_images_annotations()

pos_wins, neg_wins = [], []
for i in range(20):
    print("\rAdding img {} to test".format(i), end=' '*10)
    windows, pos = Utilities.sliding_window(im_test[i], [60]*2, 20)
    temp_pos_wins, temp_neg_wins = Utilities.build_train_set(im_test[i], anno_test[i], windows, pos)
    pos_wins.extend(temp_pos_wins)
    neg_wins.extend(temp_neg_wins)

test_set = np.concatenate([pos_wins, neg_wins], axis=0)
test_target = np.concatenate([[1]*len(pos_wins), [0]*len(neg_wins)], axis=0)#[..., None]
test_target = np.stack([test_target, -(test_target-1)], axis=1)


rd_ind = np.random.choice(test_set.shape[0], 50)

model = load_model('unet_varroas.hdf5')
pred = model.predict(test_set[rd_ind])


detection = ['mite', 'no mite']
for i in range(50):
    plt.subplot(5, 10, i+1)
    plt.imshow(test_set[rd_ind][i])
    plt.title("{},{}".format(detection[np.argmax(pred[i])], test_target[rd_ind][i]))

plt.show()























