import json
import os
import pickle
import tarfile
import xml.etree.ElementTree as ET

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import skimage
import skimage.io
from skimage.measure import block_reduce, label, regionprops
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

# ------------ Loading Images ------------

def load_images_annotations(load_x=None):
    """Loads images and annotation XML files from their directories
        - load_x: choose what files to load
            * None loads all the files and annotations except for the competition
            * 'training' -> loads training images and annotations
            * 'testing' -> loads testing images and annotations
            * 'evaluation' -> loads evaluation images and annotations
            * 'competiton' -> Loads competition images and file names for the submission"""

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

    competiton_im_path = os.path.join(DATA_PATH, 'competition/')
    im_competition_list = [os.path.join(competiton_im_path,f) for f in os.listdir(competiton_im_path) \
                        if (os.path.isfile(os.path.join(competiton_im_path, f)) and f.endswith('.jpg'))]


    train_anno_path = os.path.join(DATA_PATH,'project-data/annotations/train')
    test_anno_path = os.path.join(DATA_PATH,'project-data/annotations/test')
    validation_anno_path = os.path.join(DATA_PATH,'project-data/annotations/validation')

    if load_x == 'training':
        im_train = skimage.io.imread_collection(im_train_list)
        anno_train = [parse_file(os.path.join(train_anno_path,f)) for f in os.listdir(train_anno_path)\
                        if (os.path.isfile(os.path.join(train_anno_path, f)) and f.endswith('.xml'))]
        return im_train, anno_train

    elif load_x == 'testing':
        im_test = skimage.io.imread_collection(im_test_list)
        anno_test = [parse_file(os.path.join(test_anno_path,f)) for f in os.listdir(test_anno_path)\
                        if (os.path.isfile(os.path.join(test_anno_path, f)) and f.endswith('.xml'))]
        test_names = [f[:-4] for f in os.listdir(test_im_path) \
                        if (os.path.isfile(os.path.join(test_im_path, f)) and f.endswith('.jpg'))]            
        return im_test, anno_test, test_names

    elif load_x == 'validation':
        im_val = skimage.io.imread_collection(validation_im_path)
        anno_val = [parse_file(os.path.join(validation_anno_path,f)) for f in os.listdir(validation_anno_path)\
                        if (os.path.isfile(os.path.join(validation_anno_path, f)) and f.endswith('.xml'))]
        return im_val, anno_val

    elif load_x == 'competition':
        im_copetition = skimage.io.imread_collection(im_competition_list)
        competition_names = [f[:-4] for f in os.listdir(competiton_im_path) \
                        if (os.path.isfile(os.path.join(competiton_im_path, f)) and f.endswith('.jpg'))]
        return im_copetition, competition_names

    else:
        im_train = skimage.io.imread_collection(im_train_list)
        im_test = skimage.io.imread_collection(im_test_list)
        im_val = skimage.io.imread_collection(im_validation_list)

        anno_train = [parse_file(os.path.join(train_anno_path,f)) for f in os.listdir(train_anno_path)\
                        if (os.path.isfile(os.path.join(train_anno_path, f)) and f.endswith('.xml'))]
        anno_test = [parse_file(os.path.join(test_anno_path,f)) for f in os.listdir(test_anno_path)\
                        if (os.path.isfile(os.path.join(test_anno_path, f)) and f.endswith('.xml'))]
        anno_val = [parse_file(os.path.join(validation_anno_path,f)) for f in os.listdir(validation_anno_path)\
                        if (os.path.isfile(os.path.join(validation_anno_path, f)) and f.endswith('.xml'))]

        return im_train, im_test, im_val, anno_train, anno_test, anno_val


def build_train_set(image, annotations, windows, positions, return_pos=False):
    """Builds training or testing data as an array of windows, bias toward the negative class since there are more of those
        - image, annotations: from which extract the windows
        - windows, positions: select the windows with or with out mites in them from those
        - return_pos: labels all the input given and returns their positions (now deprecated)"""

    pos_wins,  pos_positions = [], []
    neg_wins, neg_positions  = [], []

    for i, pos in enumerate(positions):
        for anno in annotations:
            # keep the window if there is significant overlap
            if compute_IoU(pos, anno['bbox']) > 0.1:
                pos_wins.append(windows[i])
                pos_positions.append(pos)
                break
        else:
            neg_wins.append(windows[i])
            neg_positions.append(pos)

    
    if return_pos:
        return pos_wins, pos_positions, neg_wins, neg_positions
    else:
        pos_wins = np.array(pos_wins)
        neg_wins = np.array(neg_wins)
        # controls the amount of nefative class values to limit the bias on the net
        neg_wins = neg_wins[np.random.choice(neg_wins.shape[0], pos_wins.shape[0] *2)]

        return pos_wins, neg_wins


def load_set_target(load_x=None, n_images=10, window_size=[60]*2, stride=20, bootstrap=False, rand=True, plot_sample=False):
    """Creates arrays of windows to feed to the neural net for training or testing
        - load_x: see load_images_annotations
        - n_images: number of images to analyze
        - window_size, stride: parameters for the windows extraction, see sliding_windows
        - bootstrap: select n_images random images of the first n_images
        - rand: randomizes the order of the windows and lables that it outputs
        - plot_sample: plots 20 random windows with their true label"""

    if load_x is not None:
        im, anno = load_images_annotations(load_x)

    pos_wins, neg_wins = [], []

    if bootstrap:
        i_range = np.random.choice(len(im), n_images)
    else:
        i_range = range(n_images)

    for i in i_range:
        print("\rAdding imgage {} to {}".format(i, load_x), end=' '*10)
        windows, pos = sliding_window(im[i], window_size, stride)
        temp_pos_wins, temp_neg_wins = build_train_set(im[i], anno[i], windows, pos)
        pos_wins.extend(temp_pos_wins)
        neg_wins.extend(temp_neg_wins)

    # Builds a single array for the train values and the targets with one hot encoding
    set_ = np.concatenate([pos_wins, neg_wins], axis=0)
    target = np.concatenate([[1]*len(pos_wins), [0]*len(neg_wins)], axis=0)#[..., None]
    target = np.stack([target, -(target-1)], axis=1)

    # Randomization of the inputs
    if rand:
        rd_ind = np.random.choice(set_.shape[0], set_.shape[0])
        set_ = set_[rd_ind]
        target = target[rd_ind]

    if plot_sample:
        n_sample = 20
        plt.figure(figsize=(10, 10))
        window_type = ['mite', 'no mite']
        for i in range(n_sample):
            plt.subplot(4, 5, i+1)
            plt.imshow(set_[rd_ind][i])
            plt.title("{}".format(window_type[np.argmax(target[rd_ind][i])]))
            plt.axis('off')
        plt.suptitle('Ground truth over sample of windows')
        plt.show()
    print('\rDone!'+' '*50)

    return set_, target

# ------------ Utilities for the Nets ------------ #

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
    """Extracts the mites form a set of images given the annotations
        - images, annotations: uesd to extract the mites from teh given images
        - dil: dilation factor around the ground truth bbox
        - pad_value: when returning and array needs padding since the mites have different sizes
        - save: saves the file since the computation is long"""

    print("Extracting mites...")
    mites_stack = []
    for i, (img, anno_list) in enumerate(zip(images, annotations)):
        print("\r{}".format(i), end=' '*10)
        if len(anno_list) > 0:
            for anno in anno_list: 
                x, y, x_len, y_len = anno['bbox']
                # Controls the boundaries of the dilated bbox to avoid IndexError s
                x = max(0, x-int(dil*x_len))
                y = max(0, y-int(dil*y_len))
                x_len = min(x_len+int(2*dil*x_len), img.shape[1])
                y_len = min(y_len+int(2*dil*y_len), img.shape[0])
                
                mites_stack.append([img[y:y+y_len, x:x+x_len][...,0], 
                                    img[y:y+y_len, x:x+x_len][...,1], 
                                    img[y:y+y_len, x:x+x_len][...,2]])
    print("\rDone!"+' '*10)

    # Conputation of the max sizes of the bboxes for the padding
    max_size = [-1, -1]
    for mite in mites_stack:
        max_size[0] = max(max_size[0], mite[0].shape[0])
        max_size[1] = max(max_size[1], mite[0].shape[1])
        
    print("Max size of the bboxes: {}".format(max_size))

    print("Rescaling...")
    # Padding mite images
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
    """Padds given images with the given parameters
        - images: images to pad
        - pad_size: size of the padded images
        - channels in the input images 1 for grayscale 3 for rgb
        - centrering: if True the padded images are at the center of the out array
        - fiil_value: padding value
        - is_mask: different behavior for the masks of the UNet (not used)"""
    
    if is_mask:
        # Masks are stored in png for now 4 channels
        images = [img[..., 0, None] for img in images]
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

def image_pooling(images, kernel_size=(4, 4), func=np.min):
    """Pooling of the images by blocks with the given functions
        - images, image list
        - kernel_size: pooling size (3D) also used as stride
        - func: pooling function must work wint np.array()"""
    print("Pooling images...")
    pooled_images = [block_reduce(image, kernel_size, func) for image in images]
    return np.array(pooled_images)


def sliding_window(image, window_size=[100, 100], stride=None, mode=None):
    """Returns windows on the image as well as the boundaries of the slices of the image
        - image: from which to extract the windows
        - window_size: size of the window
        - stride: stride between windows same in both axis if None half of the window"""
    
    H, W, *_ = image.shape
    windows = []
    pos = []
    
    if stride is None:
        stride = window_size[0] //2

    if mode is None:
        for i in range(0, H -window_size[0], stride):
            row_s, row_e = i, (i+window_size[0])
            for j in range(0, W -window_size[1], stride):
                col_s, col_e = j ,(j+window_size[1])
                windows.append(image[row_s:row_e, col_s:col_e])               
                pos.append([row_s, col_s, *window_size])
                
    return windows, pos


def display_detection(image, windows=None, pos=None, contours=None, annotations=None):
    """
    Displays the image with other informations
        - windows are the windows where something was detected (red rectangles)
        - pos: positions (y, x, y_len, x_len) of the previous wins
        - contours (must be complex) the computed contours of the regions (blue contour)
        - annotations must correspond to the image, used to display true values (green boxes)
    """
    
    fig, ax = plt.subplots(1, figsize=(10,10))
    ax.imshow(image)

    # Displays ground truth         
    if annotations is not None:
        for anno in annotations:
            bbox = anno['bbox']
            rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3],
                    linewidth=0 , edgecolor='g', facecolor='g', alpha=0.5)
            ax.add_patch(rect)
    
    # Displays detection windows
    if windows is not None and pos is not None:
        for win, p in zip(windows, pos):
            if (win != 0).any():
                rect = patches.Rectangle( (p[1], p[0]), p[3], p[2], linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
    
    # Displays detected contours
    if contours is not None:
        for c_list in contours:
            for c in c_list:
                ax.plot(c.imag, c.real, 'b', linewidth=2)


def compute_IoU(pos1, pos2):
    """Computes the IoU criterion for 2 regions
        - pos1: positions from our functions
        - pos2: bbox from the annotations"""
    y1, x1, y1_len, x1_len = pos1
    x2, y2, y2_len, x2_len = pos2

    x_overlap = max(0, min(x1+x1_len, x2+x2_len) - max(x1, x2)+1)
    y_overlap = max(0, min(y1+y1_len, y2+y2_len) - max(y1, y2)+1)
    
    intersection = x_overlap*y_overlap
    union =  x1_len*y1_len + x2_len*y2_len -intersection     

    return intersection /union


def generate_pred_json(data, tag='baseline'):
    '''
    Input
    - data: Is a dictionary d, such that:
          d = { 
              "ID_1": [], 
              "ID_2": [[x_21, y_21, w_21, h_21], [x_22, y_22, w_22, h_22]], 
              ... 
              "ID_i": [[x_i1, y_i1, w_i1, h_i1], ..., [x_iJ, y_iJ, w_iJ, h_iJ]],
              ... 
              "ID_N": [[x_N1, y_N1, w_N1, h_N1]],
          }
          where ID is the string id of the image (e.i. 5a05e86fa07d56baef59b1cb_32.00px_1) and the value the Kx4 
          array of intergers for the K predicted bounding boxes (e.g. [[170, 120, 15, 15]])
    - tag: (optional) string that will be added to the name of the json file.
    Output
      Create a json file, "prediction_[tag].json", conatining the prediction to EvalAI format.
    '''
    unvalid_key = []
    _data = data.copy()
    for key, value in _data.items():
        try:
            # Try to convert to numpy array and cast as closest int
            print(key)
            v = np.around(np.array(value)).astype(int)
            # Check is it is a 2d array with 4 columns (x,y,w,h)
            if v.ndim != 2 or v.shape[1] != 4:
                unvalid_key.append(key)
            # Id must be a string
            if not isinstance(key, str):
                unvalid_key.append(key)
            _data[key] = v.tolist()
        # Deal with not consistant array size and empty predictions
        except (ValueError, TypeError):
            unvalid_key.append(key)
    # Remove unvalid key from dictionnary
    for key in unvalid_key: del _data[key]
    
    with open('prediction_{}.json'.format(tag), 'w') as outfile:
        json.dump(_data, outfile)


def bbox_from_net_predictions(image, windows, positions, class_predictions, annotations=None, competition=False, debug=False):

    detect_positions = []
    detect_windows = []

    # Considers only the relevant windows
    for i in range(class_predictions.shape[0]):
        if class_predictions[i] == 0:
            detect_positions.append(positions[i])
            detect_windows.append(windows[i])

    # computes the density of the output to filter the false positives
    detection = np.zeros(image.shape[:-1])
    for pos in detect_positions:
        detection[pos[0]:pos[0]+pos[2], pos[1]:pos[1]+pos[3]] += 1
    
    # Only use the regions with significant density
    thresh = max(10 ,np.percentile(detection[detection != 0].ravel(), 95)) if detection.max() > 0 else 0
    thr_detect = label((detection >= thresh))
    detected_regions = regionprops(thr_detect)

    mites_bboxes = []
    for prop in detected_regions:
        # Conditions for size and shape of the regions
        if 1*400 < prop.area <= 9*400 and max((prop.bbox[2]-prop.bbox[0]) /(prop.bbox[3]-prop.bbox[1]), (prop.bbox[3]-prop.bbox[1]) /(prop.bbox[2]-prop.bbox[0])) <= 1.6:
            # Switches the axes
            if competition:
                mites_bboxes.append([int(prop.centroid[1]-15), int(prop.centroid[0]-15), 35, 35])
            else:
                mites_bboxes.append((int(prop.centroid[0]-15), int(prop.centroid[1]-15), 35, 35))
    
    # Plots the images with predictions and truth
    if debug:
        plt.figure(figsize=(10, 10))
        plt.imshow((thr_detect != 0).astype(np.int8) *detection, cmap='viridis')
        display_detection(image, windows=detect_windows, pos=mites_bboxes, annotations=annotations)
        plt.show()

    return mites_bboxes  


def give_stats(tp, fn, fp):
    """calculates precision, recall and f1 score from true positive, false negative and false positive values"""

    try:
        precision = tp / (tp + fp)
    except ZeroDivisionError:
        precision = 0
    
    try:
        recall = tp / (tp + fn)
    except ZeroDivisionError:
        recall = 0
    
    try:
        f1 = 2 * precision * recall / (precision + recall)
    except ZeroDivisionError:
        f1 = 0
        
    return precision, recall, f1


def test_detection(bboxes, annotations):
    """Computes the true positive, false positive and false negative rates to output the precision, recall and f1_score"""

    tp, fn, fp = 0, 0, 0

    for bbox in bboxes:
        for anno in annotations:
            if compute_IoU(bbox, anno['bbox']) > 0:
                tp += 1
            else:
                fp += 1

    for anno in annotations:
        for bbox in bboxes:
            if compute_IoU(bbox, anno['bbox']) > 0:
                pass
            else:
                fn += 1 

    precision, recall, f1 = give_stats(tp, fn, fp)
    print('\r                      precision = {:.2f}, recall = {:.2f}, f1-score = {:.2f}'.format(precision, recall, f1), end=' '*20)
    return precision, recall, f1
