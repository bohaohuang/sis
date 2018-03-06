import os
import imageio
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import uabCrossValMaker
import uab_collectionFunctions
import util_functions
from bohaoCustom import uabMakeNetwork_DeepLabV2
from bohaoCustom import uabMakeNetwork_UNet

# settings
gpu = 0
batch_size = 5
input_size = [572, 572]
tile_size = [5000, 5000]
util_functions.tf_warn_level(3)

model_dir = r'/hdd6/Models/deeplab_spca/UnetCrop_spca_aug_grid_0_PS(572, 572)_BS5_EP100_LR0.0001_DS60_DR0.1_SFN32'
blCol = uab_collectionFunctions.uabCollection('spca')
blCol.readMetadata()
file_list, parent_dir = blCol.getAllTileByDirAndExt([1, 2, 3])
file_list_truth, parent_dir_truth = blCol.getAllTileByDirAndExt(0)
idx, file_list = uabCrossValMaker.uabUtilGetFolds(None, file_list, 'force_tile')
idx_truth, file_list_truth = uabCrossValMaker.uabUtilGetFolds(None, file_list_truth, 'force_tile')
# use first 5 tiles for validation
file_list_valid = uabCrossValMaker.make_file_list_by_key(
    idx, file_list, [i for i in range(250, 500)])
file_list_valid_truth = uabCrossValMaker.make_file_list_by_key(
    idx_truth, file_list_truth, [i for i in range(250, 500)])
img_mean = blCol.getChannelMeans([1, 2, 3])

# make the model
# define place holder
X = tf.placeholder(tf.float32, shape=[None, input_size[0], input_size[1], 3], name='X')
y = tf.placeholder(tf.int32, shape=[None, input_size[0], input_size[1], 1], name='y')
mode = tf.placeholder(tf.bool, name='mode')
model = uabMakeNetwork_UNet.UnetModelCrop({'X':X, 'Y':y},
                                           trainable=mode,
                                           input_size=input_size,
                                           batch_size=5, start_filter_num=32)
# create graph
model.create_graph('X', class_num=2)

# evaluate on tiles
'''model.evaluate(file_list_valid, file_list_valid_truth, parent_dir, parent_dir_truth,
               input_size, tile_size, batch_size, img_mean, model_dir, gpu,
               save_result_parent_dir='grid_vs_random', ds_name='spca',
               show_figure=True)'''

# select a patch
tile_path = r'/media/ei-edl01/data/uab_datasets/spca/data/Original_Tiles'
file_name = 'Fresno455_GT.png'
gt = imageio.imread(os.path.join(tile_path, file_name))
file_name = 'Fresno455_RGB.jpg'
img = imageio.imread(os.path.join(tile_path, file_name))

img_patch = img[0:572, 179:751, :]
gt_patch = gt[0:572, 179:751]

def img_iterator(img_patch):
    for i in range(1):
        yield np.expand_dims(img_patch, axis=0)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    model.load(model_dir, sess)
    iterator = img_iterator(img_patch)
    result = model.test('X', sess, iterator)
    result_pred = np.argmax(result, axis=3)
    result_pred_shrink = sess.run(model.pred, feed_dict={model.inputs['X']: np.expand_dims(img_patch, axis=0),
                                                         model.trainable: False})

result_pred_shrink = np.argmax(result_pred_shrink, axis=3)

plt.subplot(231)
plt.imshow(img[0:321, 179:500, :])
plt.subplot(232)
plt.imshow(gt[0:321, 179:500])
plt.subplot(234)
plt.imshow(result[0, :, :, 0])
plt.subplot(235)
plt.imshow(result[0, :, :, 1])
plt.subplot(236)
plt.imshow(result_pred[0, :, :])
plt.subplot(233)
plt.imshow(result_pred_shrink[0, :, :])
plt.show()
