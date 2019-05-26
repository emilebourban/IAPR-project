import numpy as np
# import tensorflow as tf
# import keras

import Utilities



def main():

    im_train, im_test, im_val, anno_train, anno_test, anno_val = Utilities.load_images_annotations()
    mites = Utilities.get_mites(im_train, anno_train, save=True)




if __name__ == '__main__':
    main()































