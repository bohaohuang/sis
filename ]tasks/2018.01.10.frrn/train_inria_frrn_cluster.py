"""
Created on 12/16/2017
This file show examples of following steps:
    1. Make a collection on inria with RGB data
    2. Modify the GT and map it to (0, 1)
    3. Extract patches of given size
    4. Make train and validation folds
    4. Train a UNet on those patches
"""

import os
import time
import numpy as np
import tensorflow as tf
import uabDataReader
import uabRepoPaths
import uabCrossValMaker
from bohaoCustom import uabMakeNetwork_FRRN

RunId = 0

# experiment settings
chip_size = (224, 224)
tile_size = (5000, 5000)
batch_size = 5                  # mini-batch size
learn_rate = 1e-5               # learning rate
decay_step = 60                 # learn rate dacay after 60 epochs
decay_rate=0.1                  # learn rate decay to 0.1*before
epochs=100                      # total number of epochs to rum
start_filter_num=32             # the number of filters at the first layer
n_train = 8000                  # number of samples per epoch
n_valid = 1000                  # number of samples every validation step
model_name = 'inria_aug_grid_{}'.format(RunId)   # a suffix for model name
GPU = 1                         # which gpu to use

# make network
# define place holder
X = tf.placeholder(tf.float32, shape=[None, chip_size[0], chip_size[1], 3], name='X')
y = tf.placeholder(tf.int32, shape=[None, chip_size[0], chip_size[1], 1], name='y')
mode = tf.placeholder(tf.bool, name='mode')
model = uabMakeNetwork_FRRN.FRRN({'X':X, 'Y':y},
                                 trainable=mode,
                                 model_name=model_name,
                                 input_size=chip_size,
                                 batch_size=batch_size,
                                 learn_rate=learn_rate,
                                 decay_step=decay_step,
                                 decay_rate=decay_rate,
                                 epochs=epochs,
                                 start_filter_num=start_filter_num)
model.create_graph('X', class_num=2)

# prepare data
img_mean = np.array([109.54834218, 114.86824715, 102.69644417])
patchDir_name = 'chipExtrReg_cSz{}x{}_pad{}'.format(chip_size[0], chip_size[1], model.get_overlap())
patchDir = os.path.join(uabRepoPaths.resPath, 'PatchExtr', 'inria', patchDir_name)

# make data reader
chipFiles = os.path.join(patchDir, 'fileList.txt')
# use uabCrossValMaker to get fileLists for training and validation
idx, file_list = uabCrossValMaker.uabUtilGetFolds(patchDir, 'fileList.txt', 'force_tile')
# use first 5 tiles for validation
file_list_train = uabCrossValMaker.make_file_list_by_key(idx, file_list, [i for i in range(6, 37)])
file_list_valid = uabCrossValMaker.make_file_list_by_key(idx, file_list, [i for i in range(0, 6)])

with tf.name_scope('image_loader'):
    # GT has no mean to subtract, append a 0 for block mean
    dataReader_train = uabDataReader.ImageLabelReader([3], [0, 1, 2], patchDir, file_list_train, chip_size, tile_size,
                                                      batch_size, dataAug='flip,rotate', block_mean=np.append([0], img_mean))
    # no augmentation needed for validation
    dataReader_valid = uabDataReader.ImageLabelReader([3], [0, 1, 2], patchDir, file_list_train, chip_size, tile_size,
                                                      batch_size, dataAug=' ', block_mean=np.append([0], img_mean))

# train
start_time = time.time()

model.train_config('X', 'Y', n_train, n_valid, chip_size, uabRepoPaths.modelPath, loss_type='xent')
model.run(train_reader=dataReader_train,
          valid_reader=dataReader_valid,
          pretrained_model_dir=None,        # train from scratch, no need to load pre-trained model
          isTrain=True,
          img_mean=img_mean,
          verb_step=100,                    # print a message every 100 step(sample)
          save_epoch=5,                     # save the model every 5 epochs
          gpu=GPU,
          tile_size=tile_size,
          patch_size=chip_size)

duration = time.time() - start_time
print('duration {:.2f} hours'.format(duration/60/60))
