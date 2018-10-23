import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from glob import glob
import ersa_utils
from nn import nn_utils, deeplab
from preprocess import patchExtractor
from collection import collectionMaker
from visualize import visualize_utils

patch_size = (321, 321)
overlap = 184
class_num = 2
bs = 1
suffix = 'aemo_pad'
nn_utils.set_gpu(1)

cm = collectionMaker.read_collection('aemo_pad')
chan_mean = cm.meta_data['chan_mean'][-3:]
file_list_valid = cm.load_files(field_name='aus50', field_id='', field_ext='.*rgb_hist,.*gt_d255')
pred_dir = r'/hdd/Results/aemo/deeplab_aemo_spca_hist_PS(321, 321)_BS5_EP2_LR0.001_DS50_DR0.1/default/pred'
pred_files = sorted(glob(os.path.join(pred_dir, '*.png')))

unet = deeplab.DeepLab(class_num, patch_size, suffix=suffix, batch_size=bs)
model_dir = r'/hdd6/Models/aemo/new2/deeplab_aemo_spca_hist_PS(321, 321)_BS5_EP2_LR0.001_DS50_DR0.1'
feature = tf.placeholder(tf.float32, shape=(None, patch_size[0], patch_size[1], 3))
unet.create_graph(feature)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    unet.load(model_dir, sess)
    for rgb_file, gt_file in file_list_valid:
        iou_a = []
        iou_b = []

        rgb = ersa_utils.load_file(rgb_file)
        gt = ersa_utils.load_file(gt_file)
        block = np.dstack([rgb, gt])
        grid = patchExtractor.make_grid((gt.shape[0], gt.shape[1]), patch_size, overlap)
        for patch in patchExtractor.patch_block(block, overlap // 2, grid, patch_size):
            pred = sess.run(unet.output, feed_dict={feature: np.expand_dims(patch[:, :, :3]-chan_mean, axis=0)})
            pred = np.argmax(pred[0, :, :, :], axis=-1)

            gt_patch = patch[:, :, 3]

            iou_a.append(np.sum(pred * gt_patch))
            iou_b.append(np.sum((pred + gt_patch) > 0))


        print(np.sum(iou_a), np.sum(iou_b))
