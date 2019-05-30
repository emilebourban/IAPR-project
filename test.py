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


def main():

    #%% -------------------- Conv Net -------------------- %%#

    net_filename = 'cnet_varroas_50_best.hdf5'

    TRAINING = False
    if TRAINING:
        train_set, train_target = Utilities.load_set_target('testing', 800, [50]*2, 15, bootstrap=False,  plot_sample=False)

        print("{} samples, {} channels".format(train_set.shape[0], train_set.shape[-1]))

        cnet = ConvNet(train_set.shape[1:])

        model_checkpoint = ModelCheckpoint(net_filename, monitor='loss', verbose=1, save_best_only=True)
        cnet.fit(train_set.astype(np.float32), train_target, batch_size=50, epochs=15, callbacks=[model_checkpoint])

    else:
        cnet = load_model(net_filename)

    TESTING = False
    if TESTING:
        im_test, anno_test = Utilities.load_images_annotations('training')

        detection_bboxes = []
        precision, recall, f1_score = [], [], []
        start, end = 13, 20
        for i, img in enumerate(im_test[start:end]):
            print("\rImage: {}".format(i), end=' '*20)
            windows, positions = Utilities.sliding_window(img, [50]*2, 10)
            pred = cnet.predict(np.array(windows))

            pred_class = np.argmax(pred, axis=1)

            bboxes = Utilities.bbox_from_net_predictions(img, windows, positions, pred_class, anno_test[i+start], debug=True)
            detection_bboxes.append(bboxes)
            
            if False:
                p, r, f1 = Utilities.test_detection(bboxes, anno_test[i+start])
                precision.append(p)
                recall.append(r)
                f1_score.append(f1)
        
        pickle.dump(detection_bboxes, open('detection_bboxes.pkl', mode='wb'))
        print(np.mean(precision), np.mean(recall), np.mean(f1_score))

    COMP = True
    if COMP:
        im_comp, annotations, names = Utilities.load_images_annotations('testing')
        assert len(im_comp) == len(names), '{}, {}'.format(len(im_comp),  len(names))

        comp_pred = {}

        for i, img in enumerate(im_comp):
            print("\rImage: {}".format(i), end=' '*20)
            windows, positions = Utilities.sliding_window(img, [50]*2, 10)
            pred = cnet.predict(np.array(windows))

            pred_class = np.argmax(pred, axis=1)
            bboxes = Utilities.bbox_from_net_predictions(img, windows, positions, pred_class, competition=True)

            comp_pred[names[i]] = bboxes
            print(comp_pred[names[i]])

            pickle.dump(comp_pred, open('train_predictions.pkl', mode='wb'))

    Utilities.generate_pred_json(comp_pred, tag='test')






    # im_val, anno_val = Utilities.load_images_annotations('training')
    # nb_val = 0
    # windows_val, positions_val = Utilities.sliding_window(im_val[nb_val], [60]*2, 20)

    # pred_val = np.argmax(cnet.predict(np.array(windows_val)), axis=1)
    # print(pred_val)

    # wins_detect, pos_detect = [], []
    # for i in range(len(windows_val)):
    #     if pred_val[i] == 0:
    #         wins_detect.append(windows_val[i])
    #         pos_detect.append(positions_val[i])

    # detection = np.zeros(im_val[nb_val].shape[:-1])

    # for pos in pos_detect:
    #     detection[pos[0][0]:pos[0][0]+pos[1][0], pos[0][1]:pos[0][1]+pos[1][1]] += 10
    
    # thr_detect = label((detection > 60) )
    # plt.figure(figsize=(10, 10))
    # plt.imshow(thr_detect.astype(np.int8), cmap='viridis')

    # pred_box = []
    # for prop in regionprops(thr_detect.astype(np.int8)):
    #     pred_box.append((int(prop.centroid[1]-15), int(prop.centroid[0]-15), 30, 30))

    # print(pred_box)

    # Utilities.display_detection(im_val[nb_val], windows=wins_detect, pos=pos_detect, annotations=anno_val[nb_val])
    
    plt.show()


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
    

if __name__ == '__main__':
    main()































