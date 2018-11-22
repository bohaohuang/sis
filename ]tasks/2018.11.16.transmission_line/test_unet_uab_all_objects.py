import os
import numpy as np
import tensorflow as tf
import uabUtilreader
import utils
import ersa_utils
import util_functions
import uabCrossValMaker
import uab_collectionFunctions
from bohaoCustom import uabMakeNetwork_UNet

gpu = 1
batch_size = 5
input_size = [572, 572]
util_functions.tf_warn_level(3)
model_dir = r'/hdd6/Models/aemo/infrastructure/UnetCrop_5objs_0_PS(572, 572)_BS5_EP80_LR0.0001_DS60_DR0.1_SFN32'
ds_name = 'infrastructure'

# settings
blCol = uab_collectionFunctions.uabCollection(ds_name)
blCol.readMetadata()
file_list, parent_dir = blCol.getAllTileByDirAndExt([1, 2, 3])
file_list_truth, parent_dir_truth = blCol.getAllTileByDirAndExt(0)
idx, file_list = uabCrossValMaker.uabUtilGetFolds(None, file_list, 'tile')
idx_truth, file_list_truth = uabCrossValMaker.uabUtilGetFolds(None, file_list_truth, 'tile')
# use first 5 tiles for validation
file_list_valid = uabCrossValMaker.make_file_list_by_key(idx, file_list, [1, 2, 3])
file_list_valid_truth = uabCrossValMaker.make_file_list_by_key(idx_truth, file_list_truth, [1, 2, 3])
img_mean = blCol.getChannelMeans([1, 2, 3])

# make the model
# define place holder
X = tf.placeholder(tf.float32, shape=[None, input_size[0], input_size[1], 3], name='X')
y = tf.placeholder(tf.int32, shape=[None, input_size[0], input_size[1], 1], name='y')
mode = tf.placeholder(tf.bool, name='mode')
model = uabMakeNetwork_UNet.UnetModelCrop({'X': X, 'Y': y}, trainable=mode, input_size=input_size,
                                          batch_size=batch_size, start_filter_num=32)
# create graph
model.create_graph('X', class_num=6)

# evaluate on tiles
model.evaluate(file_list_valid, file_list_valid_truth, parent_dir, parent_dir_truth,
               input_size, None, batch_size, img_mean, model_dir, gpu,
               save_result_parent_dir=ds_name, ds_name=ds_name, best_model=False,
               load_epoch_num=75, show_figure=True)
