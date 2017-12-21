"""
Created on 12/16/2017
Make a collection on inria with RGB data
Extract patches of given size
Train a ResFcn on those patches
"""

import os
import time
import numpy as np
import tensorflow as tf
import uabDataReader
import uabRepoPaths
import uabCrossValMaker
import bohaoCustom.uabPreprocClasses as bPreproc
import uabPreprocClasses
import uab_collectionFunctions
import uab_DataHandlerFunctions
from bohaoCustom import uabMakeNetwork_ResFCN

# experiment settings
chip_size = (224, 224)
tile_size = (5000, 5000)
batch_size = 5
n_train = 8000
n_valid = 1000
model_name = 'inria_fr_mr'
IMG_MEAN = np.array((109.629784946, 114.94964751, 102.778073453), dtype=np.float32)
GPU = 1

# make network
# define place holder
X = tf.placeholder(tf.float32, shape=[None, chip_size[0], chip_size[1], 3], name='X')
y = tf.placeholder(tf.int32, shape=[None, chip_size[0], chip_size[1], 1], name='y')
mode = tf.placeholder(tf.bool, name='mode')
model = uabMakeNetwork_ResFCN.ResFcnModel({'X':X, 'Y':y},
                                          trainable=mode,
                                          input_size=chip_size,
                                          batch_size=5)
model.create_graph('X', class_num=2)

# create collection
# the original file is in /ei-edl01/data/uab_datasets/inria
blCol = uab_collectionFunctions.uabCollection('inria')
opDetObj = bPreproc.uabOperTileDivide(255)
rescObj = uabPreprocClasses.uabPreprocMultChanOp([], 'GT_Divide.tif', 'Map GT to (0, 1)', [3], opDetObj)
rescObj.run(blCol)
print(blCol.readMetadata())

# extract patches
extrObj = uab_DataHandlerFunctions.uabPatchExtr([0, 1, 2, 4], # extract all 4 channels
                                                cSize=chip_size, # patch size as 224*224
                                                numPixOverlap=0,  # overlap as 0
                                                extSave=['jpg', 'jpg', 'jpg', 'png'], # save rgb files as jpg and gt as png
                                                isTrain=True,
                                                gtInd=3)
patchDir = extrObj.run(blCol)

# make data reader
chipFiles = os.path.join(patchDir, 'fileList.txt')
# use uabCrossValMaker to get fileLists for training and validation
idx, file_list = uabCrossValMaker.uabUtilGetFolds(patchDir, 'fileList.txt', 'force_tile')
file_list_train = uabCrossValMaker.make_file_list_by_key(idx, file_list, [i for i in range(6, 37)])
file_list_valid = uabCrossValMaker.make_file_list_by_key(idx, file_list, [i for i in range(0, 6)])

with tf.name_scope('image_loader'):
    dataReader_train = uabDataReader.ImageLabelReader([3], [0, 1, 2], patchDir, file_list_train, chip_size, tile_size,
                                                      batch_size, dataAug='flip,rotate', block_mean=np.append([0], IMG_MEAN))
    dataReader_valid = uabDataReader.ImageLabelReader([3], [0, 1, 2], patchDir, file_list_train, chip_size, tile_size,
                                                      batch_size, dataAug=' ', block_mean=np.append([0], IMG_MEAN))

# train
start_time = time.time()

model.train_config('X', 'Y', n_train, n_valid, chip_size, uabRepoPaths.modelPath, loss_type='xent')
model.run(train_reader=dataReader_train,
          valid_reader=dataReader_valid,
          pretrained_model_dir=None,
          isTrain=True,
          img_mean=IMG_MEAN,
          verb_step=100,
          save_epoch=5,
          gpu=GPU,
          tile_size=tile_size,
          patch_size=chip_size)

duration = time.time() - start_time
print('duration {:.2f} hours'.format(duration/60/60))
