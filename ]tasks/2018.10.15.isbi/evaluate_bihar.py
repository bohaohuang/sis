import os
import time
import imageio
import numpy as np
import tensorflow as tf
import util_functions
import uab_collectionFunctions
from bohaoCustom import uabMakeNetwork_UNet

gpu = 0
batch_size = 1
input_size = [316, 316]
util_functions.tf_warn_level(3)
ds_name = 'bihar_building'

# settings
blCol = uab_collectionFunctions.uabCollection(ds_name)
blCol.readMetadata()
img_mean = blCol.getChannelMeans([0, 1, 2])

# make the model
# define place holder
model_dir = r'/hdd6/Models/bihar_building/UnetCrop_bihar_building_0_PS(316, 316)_BS5_EP50_LR0.0001_DS30_DR0.1_SFN32'
X = tf.placeholder(tf.float32, shape=[None, input_size[0], input_size[1], 3], name='X')
y = tf.placeholder(tf.int32, shape=[None, input_size[0], input_size[1], 1], name='y')
mode = tf.placeholder(tf.bool, name='mode')
model = uabMakeNetwork_UNet.UnetModelCrop({'X': X, 'Y': y}, trainable=mode, input_size=input_size,
                                          batch_size=batch_size, start_filter_num=32)
# create graph
model.create_graph('X', class_num=2)

# walk through folders

