import tarfile
import os
import skimage
import skimage.io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pickle

import xml.etree.ElementTree as ET
from skimage.transform import resize

DATA_PATH = './data/'

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
    train_im_path = os.path.join(DATA_PATH,'project-data/images/train')
    im_train_list = [os.path.join(train_im_path,f) for f in os.listdir(train_im_path)\
                    if (os.path.isfile(os.path.join(train_im_path, f)) and f.endswith('.jpg'))]

    #load test images
    test_im_path = os.path.join(DATA_PATH,'project-data/images/test')
    im_test_list = [os.path.join(test_im_path,f) for f in os.listdir(test_im_path) \
                    if (os.path.isfile(os.path.join(test_im_path, f)) and f.endswith('.jpg'))]

    #load validation images
    validation_im_path = os.path.join(DATA_PATH,'project-data/images/validation')
    im_validation_list = [os.path.join(validation_im_path,f) for f in os.listdir(validation_im_path) \
                        if (os.path.isfile(os.path.join(validation_im_path, f)) and f.endswith('.jpg'))]

    im_train = skimage.io.imread_collection(im_train_list)
    im_test = skimage.io.imread_collection(im_test_list)
    im_val = skimage.io.imread_collection(im_validation_list)

    #load train annotations
    train_anno_path = os.path.join(DATA_PATH,'project-data/annotations/train')
    anno_train = [parse_file(os.path.join(train_anno_path,f)) for f in os.listdir(train_anno_path)\
                    if (os.path.isfile(os.path.join(train_anno_path, f)) and f.endswith('.xml'))]

    #load test annotations
    test_anno_path = os.path.join(DATA_PATH,'project-data/annotations/test')
    anno_test = [parse_file(os.path.join(test_anno_path,f)) for f in os.listdir(test_anno_path)\
                    if (os.path.isfile(os.path.join(test_anno_path, f)) and f.endswith('.xml'))]

    #loaf validation annotations
    validation_anno_path = os.path.join(DATA_PATH,'project-data/annotations/validation')
    anno_val = [parse_file(os.path.join(validation_anno_path,f)) for f in os.listdir(validation_anno_path)\
                    if (os.path.isfile(os.path.join(validation_anno_path, f)) and f.endswith('.xml'))]

    return im_train, im_test, im_val, anno_train, anno_test, anno_val


def load_UNet_masks(dir=DATA_PATH):
    """Loads the masks for the training of a UNet"""
    
    dirs = [os.path.join(dir, 'masks/', folder) for folder in ['image_tr', 'image_te', 'image_val']]
    filenames = [[], [], []]

    for i, dirpath in enumerate(dirs):
        filenames[i] = [os.path.join(dirpath, name) for name in os.listdir(dirpath) if name.endswith('.png')]    

    mask_tr = skimage.io.imread_collection(filenames[0])
    mask_te = skimage.io.imread_collection(filenames[1])
    mask_val = skimage.io.imread_collection(filenames[2])

    return mask_tr, mask_te, mask_val


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
        filepath = os.path.join(DATA_PATH, 'extracted_mites.pkl')
        pickle.dump( rescaled_mites_stack.astype(np.int16), open(filepath, "wb" ) )

    return rescaled_mites_stack.astype(np.int16)


def build_UNet_mask(images, annotations, save=False, savedir=None, filenames=None):
    """Computes masks of the images for the training and testing of a UNet"""

    def segment_mite(window):
        """TODO: improve to mite shape"""
        segmentation = np.ones_like(window[...,0])
        return segmentation

    masks = []
    dil = 0
    print("Building masks...")
    for i, (img, anno_list) in enumerate(zip(images, annotations)):
        # Creates a binary mask of the image, 1 channel
        mask = np.zeros_like(img[...,0])
        print("\rImage: {}".format(i), end=' '*10)
        if len(anno_list) > 0:
            for anno in anno_list: 
                x, y, x_len, y_len = anno['bbox']
                # Conditions to keep segmentation in the image
                x = max(0, x-int(dil*x_len))
                y = max(0, y-int(dil*y_len))
                x_len = min(x_len+int(2*dil*x_len), img.shape[1])
                y_len = min(y_len+int(2*dil*y_len), img.shape[0])

                mask[y:y+y_len, x:x+x_len] = segment_mite(img[y:y+y_len, x:x+x_len])
        
        masks.append(mask)

    print("\rDone!"+' '*10)    

    if save and savedir is not None:
        if filenames is None:
            print("Saving...")
            for i, mask in enumerate(masks):                
                filename = 'image_val_{}.png'.format(i)
                print("\rSaving: {}".format(filename), end=' '*10)
                plt.imsave(os.path.join(savedir, filename), mask, cmap='gray')
            print("\rDone!"+' '*50)
        else:
            pass


def pad_images(images, pad_size=(2000, 2000), channels=None, centering=False, fill_value=0, is_mask=False):
    
    if is_mask:
        # Masks are stored in png for now 4 channels
        print(images[0].shape)
        images = [img[..., 0, None] for img in images]
        print(images[0].shape)
    else:
        if channels == 1:
            print("\rGrayscaling...", end='')
            images = [skimage.color.rgb2gray(img)[..., None] for img in images]
        elif channels == 3 or channels is None:
            pass
        else:
            raise NotImplementedError("channels must be 1 or 3")

    n_channels = channels if channels is not None else images[0].shape[-1]
    padded_array = np.full((len(images), *pad_size, n_channels), fill_value, dtype=np.uint8)

    if centering:
        for i, img in enumerate(images):
            print("\rPadding image: {}".format(i), end=' '*10)
            y_len, x_len = img.shape[:2]
            y_id, x_id = (pad_size[0] -y_len) //2, (pad_size[1] -x_len) //2
            padded_array[i, y_id:y_id+y_len, x_id:x_id+x_len, :] = img
        print("\rDone"+' '*20)
    else:
        for i, img in enumerate(images):
            print("\rPadding image: {}".format(i), end=' '*10)
            y_len, x_len = img.shape[:2]
            padded_array[i, :y_len, :x_len, :] = img
        print("\rDone"+' '*20)
    
    return padded_array


def resize_images(images, out_size=(256, 256), channels=1, AA=False):
    #TODO return array
    if channels == 1:
        print("\rGrayscaling...", end=' '*10)
        images = [skimage.color.rgb2gray(img) for img in images]

    for i, image in enumerate(images):
        print("\rResizing image: {}".format(i), end=' '*10)
        resized = [resize(img, out_size, anti_aliasing=AA) for img in images]
    print("\rDone"+' '*20)

    return resized



















