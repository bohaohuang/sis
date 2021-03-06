# this file comes from https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
#
# Cityscapes labels
#

import numpy as np
from nn import nn_utils
from collections import namedtuple


#--------------------------------------------------------------------------------
# Definitions
#--------------------------------------------------------------------------------

# a label and all meta information
Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.

    'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
                    # ground truth images with train IDs, using the tools provided in the
                    # 'preparation' folder. However, make sure to validate or submit results
                    # to our evaluation server using the regular IDs above!
                    # For trainIds, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.
                    # Max value is 255!

    'category'    , # The name of the category that this label belongs to

    'categoryId'  , # The ID of this category. Used to create ground truth images
                    # on category level.

    'hasInstances', # Whether this label distinguishes between single instances or not

    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations or not

    'color'       , # The color of this label
    ] )


#--------------------------------------------------------------------------------
# A list of all labels
#--------------------------------------------------------------------------------

# Please adapt the train IDs as appropriate for your approach.
# Note that you might want to ignore labels with ID 255 during training.
# Further note that the current train IDs are only a suggestion. You can use whatever you like.
# Make sure to provide your results using the original IDs and not the training IDs.
# Note that many IDs are ignored in evaluation and thus you never need to predict these!

labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
    Label(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
    Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
    Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
    Label(  'parking'              ,  9 ,      255 , 'flat'            , 1       , False        , True         , (250,170,160) ),
    Label(  'rail track'           , 10 ,      255 , 'flat'            , 1       , False        , True         , (230,150,140) ),
    Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
    Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
    Label(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
    Label(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
    Label(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
    Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
    Label(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
    Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
    Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
    Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
    Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
    Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
    Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
    Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
    Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
    Label(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
    Label(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
    Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
    Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
    Label(  'license plate'        , -1 ,       -1 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
]


def decode_labels(label, label_num=2):
    """
    Decode label prediction map into rgb color map
    :param label: label prediction map
    :param label_num: #distinct classes in ground truth
    :return:
    """
    label_colors = get_label_color_dict()
    if len(label.shape) == 4:
        n, h, w, c = label.shape
        outputs = np.zeros((n, h, w, 3), dtype=np.uint8)
        for i in range(n):
            pixels = np.zeros((h, w, 3), dtype=np.uint8)
            for j in range(h):
                for k in range(w):
                    pixels[j, k] = label_colors[np.int(label[i, j, k, 0])]
            outputs[i] = pixels
    elif len(label.shape) == 3:
        h, w, c = label.shape
        outputs = np.zeros((h, w, 3), dtype=np.uint8)
        for j in range(h):
            for k in range(w):
                outputs[j, k] = label_colors[np.int(label[j, k, 0])]
    else:
        h, w = label.shape
        outputs = np.zeros((h, w, 3), dtype=np.uint8)
        for j in range(h):
            for k in range(w):
                outputs[j, k] = label_colors[np.int(label[j, k])]
    return outputs


def image_summary(image, truth, prediction, img_mean=np.array((0, 0, 0), dtype=np.float32), label_num=2):
    """
    Make a image summary where the format is image|truth|pred
    :param image: input rgb image
    :param truth: ground truth
    :param prediction: network prediction
    :param img_mean: image mean, need to add back here for visualization
    :param label_num: #distinct classes in ground truth
    :return:
    """
    truth_img = decode_labels(truth, label_num)

    prediction = nn_utils.pad_prediction(image, prediction)

    pred_labels = nn_utils.get_pred_labels(prediction)
    pred_img = decode_labels(pred_labels, label_num)
    _, h, w, _ = image.shape
    if w / h > 1.5:
        # concatenate image horizontally if it is too wide
        return np.concatenate([image + img_mean, truth_img, pred_img], axis=1)
    else:
        return np.concatenate([image + img_mean, truth_img, pred_img], axis=2)


def get_label_color_dict():
    label_colors = {}
    for l in labels:
        label_colors[l.trainId] = l.color
    return label_colors


def get_label_color_dict_regular():
    label_colors = {}
    for l in labels:
        label_colors[l.id] = l.color
    return label_colors


def decode_labels_regular(label):
    """
    Decode label prediction map into rgb color map
    :param label: label prediction map
    :param label_num: #distinct classes in ground truth
    :return:
    """
    label_colors = get_label_color_dict_regular()
    h, w = label.shape
    outputs = np.zeros((h, w, 3), dtype=np.uint8)
    for j in range(h):
        for k in range(w):
            outputs[j, k] = label_colors[np.int(label[j, k])]
    return outputs
