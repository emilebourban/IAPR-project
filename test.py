import numpy as np
import matplotlib.pyplot as plt
import skimage
import pickle
# import tensorflow as tf
# import keras

import Utilities
from model import *

LOAD_MITES = False
BUILD_MASKS = False


def main():

    im_train, im_test, im_val, anno_train, anno_test, anno_val = Utilities.load_images_annotations()
    mites = Utilities.get_mites(im_train, anno_train, save=True) if LOAD_MITES else pickle.load(open('data/extracted_mites.pkl', mode='rb'))

    if BUILD_MASKS:
        Utilities.build_UNet_mask(im_train, anno_train, save=True, savedir='data/masks/image_tr/')
        Utilities.build_UNet_mask(im_test, anno_test, save=True, savedir='data/masks/image_te/')
        Utilities.build_UNet_mask(im_val, anno_val, save=True, savedir='data/masks/image_val/')
    else:
        mask_tr, mask_te, mask_val = Utilities.load_UNet_masks()

    # train_set = np.expand_dims(np.array(Utilities.resize_images(im_test[:50], (256, 256), 1)), -1)
    # train_target = np.expand_dims(np.array(Utilities.resize_images(mask_te[:50], (256, 256), 1)), -1)

    train_set = Utilities.pad_images(im_test[:100], (2000, 2000), fill_value=0, channels=3, centering=True)
    train_target = Utilities.pad_images(mask_te[:100], (2000, 2000), fill_value=0, is_mask=True, centering=True)

    plt.figure(figsize=(10, 10))

    for i in range(9):
        plt.subplot(3, 3, i+1)
        plt.imshow(train_target[i][...,0])

    plt.show()


    # Creating the U Net
    unet = smaller_UNet(input_size=train_set.shape[1:])
    unet.summary()

    model_checkpoint = ModelCheckpoint('unet_varroas.hdf5', monitor='loss', verbose=1, save_best_only=True)
    unet.fit(train_set, train_target, batch_size=5, epochs=3, callbacks=[model_checkpoint])

    # pred = unet.predict(np.expand_dims(np.array(Utilities.resize_images([mask_tr[1]], (256, 256), 1)), -1)[0])
    # plt.imshow(pred)
    # plt.show()


if __name__ == '__main__':
    main()































