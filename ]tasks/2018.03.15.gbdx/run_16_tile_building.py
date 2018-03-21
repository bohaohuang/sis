import os
import tensorflow as tf
from glob import glob
import uab_collectionFunctions
import util_functions
from bohaoCustom import uabMakeNetwork_DeepLabV2

# settings
gpu = 0
batch_size = 5
input_size = [736, 736]
tile_size = [2541, 2541]
util_functions.tf_warn_level(3)
file_dir = r'/media/ei-edl01/data/uab_datasets/sp/DATA_BUILDING_AND_PANEL'

tf.reset_default_graph()

model_dir = r'/hdd6/Models/DeepLab_rand_grid/DeeplabV3_res101_inria_aug_grid_0_PS(321, 321)_BS5_EP100_LR1e-05_DS40_DR0.1_SFN32'
blCol = uab_collectionFunctions.uabCollection('spca')
blCol.readMetadata()
# use first 5 tiles for validation
file_list_valid = [[os.path.basename(x)] for x in sorted(glob(os.path.join(file_dir, '*.jpg')))]
file_list_valid_truth = [os.path.basename(x) for x in sorted(glob(os.path.join(file_dir, '*.png')))]
img_mean = blCol.getChannelMeans([1, 2, 3])

# make the model
# define place holder
X = tf.placeholder(tf.float32, shape=[None, input_size[0], input_size[1], 3], name='X')
y = tf.placeholder(tf.int32, shape=[None, input_size[0], input_size[1], 1], name='y')
mode = tf.placeholder(tf.bool, name='mode')
model = uabMakeNetwork_DeepLabV2.DeeplabV3({'X':X, 'Y':y},
                                           trainable=mode,
                                           input_size=input_size,
                                           batch_size=5, start_filter_num=32)
# create graph
model.create_graph('X', class_num=2)

# evaluate on tiles
model.evaluate(file_list_valid, file_list_valid_truth, file_dir, file_dir,
               input_size, tile_size, batch_size, img_mean, model_dir, gpu,
               save_result_parent_dir='gbdx_cmp', ds_name='sp')
