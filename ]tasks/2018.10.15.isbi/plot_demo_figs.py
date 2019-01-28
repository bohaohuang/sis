import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from glob import glob
from natsort import natsorted
import utils
import ersa_utils
import uab_collectionFunctions
from visualize import visualize_utils
from bohaoCustom import uabMakeNetwork_DeepLabV2

# settings
data_dir = r'/media/ei-edl01/data/uab_datasets/bihar_building/data/Original_Tiles'
input_size = (300, 300)
batch_size = 1

# set gpu to use
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISISBLE_DEVICES'] = '0'

# settings
blCol = uab_collectionFunctions.uabCollection('bihar_building')
blCol.readMetadata()
img_mean = blCol.getChannelMeans([0, 1, 2])

# make the model
# define place holder
model_dir = r'/media/ei-edl01/user/bh163/models/bihar_building2'
X = tf.placeholder(tf.float32, shape=[None, input_size[0], input_size[1], 3], name='X')
y = tf.placeholder(tf.int32, shape=[None, input_size[0], input_size[1], 1], name='y')
mode = tf.placeholder(tf.bool, name='mode')
model = uabMakeNetwork_DeepLabV2.DeeplabV3({'X': X, 'Y': y}, trainable=mode, input_size=input_size,
                                           batch_size=batch_size, start_filter_num=32)
# create graph
model.create_graph('X', class_num=2)

rgbs = natsorted(glob(os.path.join(data_dir, '*.tif')))
gts = natsorted(glob(os.path.join(data_dir, '*.png')))

img_dir, task_dir = utils.get_task_img_folder()
save_dir = os.path.join(img_dir, 'bihar_patch_preds')
ersa_utils.make_dir_if_not_exist(save_dir)

with tf.Session() as sess:
    model.load(model_dir, sess)
    for rgb_file, gt_file in zip(rgbs, gts):
        rgb = ersa_utils.load_file(rgb_file)
        gt = ersa_utils.load_file(gt_file)

        X_batch = np.expand_dims(rgb-img_mean, axis=0)
        pred = sess.run(model.output, feed_dict={model.inputs['X']: X_batch,
                                                 model.trainable: False})
        pred = np.argmax(pred[0, :, :, :], axis=-1)

        save_name = os.path.join(save_dir, 'cmp_{}.png'.format(os.path.basename(rgb_file)[:-4]))
        visualize_utils.compare_figures([rgb.astype(np.uint8), gt, pred], (1, 3), fig_size=(15, 5), show_fig=False)
        plt.savefig(save_name)
        plt.close()
