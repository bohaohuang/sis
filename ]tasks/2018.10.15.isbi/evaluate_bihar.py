import os
import time
import imageio
import numpy as np
import tensorflow as tf
import util_functions
import uab_collectionFunctions
from bohaoCustom import uabMakeNetwork_DeepLabV2

gpu = 0
batch_size = 1
input_size = [300, 300]
util_functions.tf_warn_level(3)
ds_name = 'bihar_building'

# settings
blCol = uab_collectionFunctions.uabCollection(ds_name)
blCol.readMetadata()
img_mean = blCol.getChannelMeans([0, 1, 2])

# make the model
# define place holder
model_dir = r'/hdd6/Models/bihar_building/DeeplabV3_bihar_building_0_PS(300, 300)_BS5_EP10_LR1e-08_DS1_DR0.8_SFN32'
X = tf.placeholder(tf.float32, shape=[None, input_size[0], input_size[1], 3], name='X')
y = tf.placeholder(tf.int32, shape=[None, input_size[0], input_size[1], 1], name='y')
mode = tf.placeholder(tf.bool, name='mode')
model = uabMakeNetwork_DeepLabV2.DeeplabV3({'X': X, 'Y': y}, trainable=mode, input_size=input_size,
                                           batch_size=batch_size, start_filter_num=32)
# create graph
model.create_graph('X', class_num=2)

# walk through folders
import ersa_utils
fig = r'/media/ei-edl01/data/uab_datasets/bihar_building/data/Original_Tiles/i_9600_6600_RGB.tif'
fig = ersa_utils.load_file(fig)

model.config = tf.ConfigProto(allow_soft_placement=True)
with tf.Session(config=model.config) as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    model.load(model_dir, sess, epoch=None, best_model=False)

    input_ = np.expand_dims(fig - img_mean, axis=0)

    pred = sess.run(model.output, feed_dict={model.inputs['X']: input_,
                                             model.trainable: False})

    pred = np.argmax(pred[0, :, :, :], axis=-1)

    from visualize import visualize_utils
    visualize_utils.compare_figures([fig, pred], (1, 2), fig_size=(12, 5))
