import tarfile
import os
import skimage
# import skimage.io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pickle

import xml.etree.ElementTree as ET

DATA_PATH = './data/project-data'

def parse_file(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(float(bbox.find('xmin').text)),
                              int(float(bbox.find('ymin').text)),
                              int(float(bbox.find('xmax').text))-int(float(bbox.find('xmin').text)),
                              int(float(bbox.find('ymax').text))-int(float(bbox.find('ymin').text))]
        objects.append(obj_struct)

    return objects


def load_images_annotations():

    #load train images
    train_im_path = os.path.join(DATA_PATH,'images/train')
    im_train_list = [os.path.join(train_im_path,f) for f in os.listdir(train_im_path)\
                    if (os.path.isfile(os.path.join(train_im_path, f)) and f.endswith('.jpg'))]

    #load test images
    test_im_path = os.path.join(DATA_PATH,'images/test')
    im_test_list = [os.path.join(test_im_path,f) for f in os.listdir(test_im_path) \
                    if (os.path.isfile(os.path.join(test_im_path, f)) and f.endswith('.jpg'))]

    #load validation images
    validation_im_path = os.path.join(DATA_PATH,'images/validation')
    im_validation_list = [os.path.join(validation_im_path,f) for f in os.listdir(validation_im_path) \
                        if (os.path.isfile(os.path.join(validation_im_path, f)) and f.endswith('.jpg'))]

    im_train = skimage.io.imread_collection(im_train_list)
    im_test = skimage.io.imread_collection(im_test_list)
    im_val = skimage.io.imread_collection(im_validation_list)

    #load train annotations
    train_anno_path = os.path.join(DATA_PATH,'annotations/train')
    anno_train = [parse_file(os.path.join(train_anno_path,f)) for f in os.listdir(train_anno_path)\
                    if (os.path.isfile(os.path.join(train_anno_path, f)) and f.endswith('.xml'))]

    #load test annotations
    test_anno_path = os.path.join(DATA_PATH,'annotations/test')
    anno_test = [parse_file(os.path.join(test_anno_path,f)) for f in os.listdir(test_anno_path)\
                    if (os.path.isfile(os.path.join(test_anno_path, f)) and f.endswith('.xml'))]

    #loaf validation annotations
    validation_anno_path = os.path.join(DATA_PATH,'annotations/validation')
    anno_val = [parse_file(os.path.join(validation_anno_path,f)) for f in os.listdir(validation_anno_path)\
                    if (os.path.isfile(os.path.join(validation_anno_path, f)) and f.endswith('.xml'))]

    return im_train, im_test, im_val, anno_train, anno_test, anno_val


def get_mites(images, annotations, dil=0.3, pad_val=np.nan, save=False):
    """Extracts the mites form a set of images given the annotations"""

    print("Extracting mites...")
    mites_stack = []
    for i, (img, anno_list) in enumerate(zip(images, annotations)):
        print("\r{}".format(i), end=' '*10)
        if len(anno_list) > 0:
            for anno in anno_list: 
                x, y, x_len, y_len = anno['bbox']
                x = max(0, x-int(dil*x_len))
                y = max(0, y-int(dil*y_len))
                x_len = min(x_len+int(2*dil*x_len), img.shape[1])
                y_len = min(y_len+int(2*dil*y_len), img.shape[0])
                
                mites_stack.append([img[y:y+y_len, x:x+x_len][...,0], 
                                    img[y:y+y_len, x:x+x_len][...,1], 
                                    img[y:y+y_len, x:x+x_len][...,2]])
    print("\rDone!"+' '*10)

    max_size = [-1, -1]
    for mite in mites_stack:
        max_size[0] = max(max_size[0], mite[0].shape[0])
        max_size[1] = max(max_size[1], mite[0].shape[1])
        
    print("Max size of the bboxes: {}".format(max_size))

    print("Rescaling...")
    rescaled_mites_stack = np.full((len(mites_stack), 90, 90, 3), pad_val)
    for i, mite in enumerate(mites_stack):
        print("\r{}".format(i), end=' '*10)
        y, x = mite[0].shape
        st_y, st_x = int(rescaled_mites_stack.shape[1] //2 -y//2), int(rescaled_mites_stack.shape[2] //2 -x//2)
        rescaled_mites_stack[i,st_y:st_y+y, st_x:st_x+x,0] = mite[0] 
        rescaled_mites_stack[i,st_y:st_y+y, st_x:st_x+x,1] = mite[1]
        rescaled_mites_stack[i,st_y:st_y+y, st_x:st_x+x,2] = mite[2]        
                                        
    print("\rDone!"+' '*10)

    if save:
        filepath = os.path.join('data/', 'extracted_mites.pkl')
        pickle.dump( rescaled_mites_stack, open(filepath, "wb" ) )

    return rescaled_mites_stack.astype(np.int16)




















