import os
import sys
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


def load_train_data(n_images, rand=True, plot_sample=False):
    im_train, anno_train = Utilities.load_images_annotations('training')

    pos_wins, neg_wins = [], []
    for i in range(50):
        print("\rAdding imgage {} to training".format(i), end=' '*10)
        windows, pos = Utilities.sliding_window(im_train[i], [60]*2, 20)
        temp_pos_wins, temp_neg_wins = Utilities.build_train_set(im_train[i], anno_train[i], windows, pos)
        pos_wins.extend(temp_pos_wins)
        neg_wins.extend(temp_neg_wins)

    train_set = np.concatenate([pos_wins, neg_wins], axis=0)
    train_target = np.concatenate([[1]*len(pos_wins), [0]*len(neg_wins)], axis=0)#[..., None]
    train_target = np.stack([train_target, -(train_target-1)], axis=1)

    # Randomization of the inputs
    if rand:
        rd_ind = np.random.choice(train_set.shape[0], train_set.shape[0])
        train_set = train_set[rd_ind]
        train_target = train_target[rd_ind]


    if plot_sample:
        n_sample = 20
        plt.figure(figsize=(10, 10))
        window_type = ['mite', 'no mite']
        for i in range(n_sample):
            plt.subplot(4, 5, i+1)
            plt.imshow(train_set[rd_ind][i])
            plt.title("{}".format(window_type[np.argmax(train_target[rd_ind][i])]))
            plt.axis('off')
        plt.suptitle('Ground truth over sample of window')
        plt.show()

    return train_set, train_target


def main():

    im_train, im_test, im_val, anno_train, anno_test, anno_val = Utilities.load_images_annotations()

    # mites = Utilities.get_mites(im_train, anno_train, save=True) if LOAD_MITES else pickle.load(open('data/extracted_mites.pkl', mode='rb'))

    # if BUILD_MASKS:
    #     Utilities.build_UNet_mask(im_train, anno_train, save=True, savedir='data/masks/image_tr/')
    #     Utilities.build_UNet_mask(im_test, anno_test, save=True, savedir='data/masks/image_te/')
    #     Utilities.build_UNet_mask(im_val, anno_val, save=True, savedir='data/masks/image_val/')
    # else:
    #     mask_tr, mask_te, mask_val = Utilities.load_UNet_masks()

    # train_set = Utilities.pad_images(im_test[:500], (2048, 2048), fill_value=0, channels=3, centering=True)
    # train_target = Utilities.pad_images(mask_te[:500], (2048, 2048), fill_value=0, is_mask=True, centering=True)

    # # Reduce dimension of the images to 256x256
    # train_set = Utilities.image_pooling(train_set, (8, 8, 1), np.min)
    # train_target = Utilities.image_pooling(train_target, (8, 8, 1), np.max)

    # print(train_set.dtype, train_set.shape)
    # print(sys.getsizeof(train_set), sys.getsizeof(train_target))

    # # Creating the U Net
    # unet = smaller_UNet(input_size=(256, 256, 3))#train_set.shape[:, :256, :256, :][1:])
    # unet.summary()

    # model_checkpoint = ModelCheckpoint('unet_varroas.hdf5', monitor='loss', verbose=1, save_best_only=True)
    # unet.fit(train_set, train_target, batch_size=10, epochs=5, callbacks=[model_checkpoint])






    # use_nbr = 771
    # use_img, use_anno = im_train[use_nbr], anno_train[use_nbr]

    # windows, pos = Utilities.sliding_window(use_img, [60]*2, 20)

    # plot_wins = True
    # if plot_wins:
    #     pos_windows, pos_positions, neg_windows, neg_positions = Utilities.build_train_set(use_img, use_anno, windows, pos, return_pos=True)

    #     Utilities.display_detection(use_img, pos_windows, pos_positions, annotations=use_anno)
    #     plt.show()

    # pos_wins, neg_wins = [], []
    # for i in range(50):
    #     print("\rAdding imgage {} to training".format(i), end=' '*10)
    #     windows, pos = Utilities.sliding_window(im_train[i], [60]*2, 20)
    #     temp_pos_wins, temp_neg_wins = Utilities.build_train_set(im_train[i], anno_train[i], windows, pos)
    #     pos_wins.extend(temp_pos_wins)
    #     neg_wins.extend(temp_neg_wins)

    # train_set = np.concatenate([pos_wins, neg_wins], axis=0)
    # train_target = np.concatenate([[1]*len(pos_wins), [0]*len(neg_wins)], axis=0)#[..., None]
    # train_target = np.stack([train_target, -(train_target-1)], axis=1)



    train_set, train_target = load_train_data(50, plot_sample=True)

    print(train_set.shape, train_target.shape)

    cnet = ConvNet(train_set.shape[1:])

    model_checkpoint = ModelCheckpoint('cnet_varroas.hdf5', monitor='loss', verbose=1, save_best_only=True)
    cnet.fit(train_set.astype(np.float32), train_target, batch_size=20, epochs=5, callbacks=[model_checkpoint])
    

if __name__ == '__main__':
    main()































