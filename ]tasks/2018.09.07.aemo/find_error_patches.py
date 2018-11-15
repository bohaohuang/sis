import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from glob import glob
from skimage import measure
import utils
import ersa_utils
import uabDataReader
import uabUtilreader
import util_functions
import uabCrossValMaker
import uab_collectionFunctions
from nn import nn_utils
from visualize import visualize_utils
from bohaoCustom import uabMakeNetwork_UNet

gpu = -1
nn_utils.set_gpu(gpu)
batch_size = 5
input_size = [572, 572]
tile_size = [5000, 5000]
util_functions.tf_warn_level(3)

model_dir = r'/hdd6/Models/aemo/aemo/UnetCrop_aemo_ft_1_PS(572, 572)_BS5_EP80_LR0.001_DS30_DR0.1_SFN32'
ds_name = 'aemo'
img_dir, task_dir = utils.get_task_img_folder()
save_dir = os.path.join(img_dir, 'hard_samples_reweight')
#save_dir = os.path.join(img_dir, 'hard_samples_demo')
ersa_utils.make_dir_if_not_exist(save_dir)
f = open(os.path.join(save_dir, 'file_list.txt'), 'w+')

# settings
blCol = uab_collectionFunctions.uabCollection(ds_name)
img_mean = blCol.getChannelMeans([1, 2, 3])
parent_dir = r'/home/lab/Documents/bohao/data/aemo/aemo_align'

rgb_list = sorted(glob(os.path.join(parent_dir, '*rgb.tif')))[:-2]
gt_list = sorted(glob(os.path.join(parent_dir, '*gt_d255.tif')))[:-2]

# make the model
# define place holder
X = tf.placeholder(tf.float32, shape=[None, input_size[0], input_size[1], 3], name='X')
y = tf.placeholder(tf.int32, shape=[None, input_size[0], input_size[1], 1], name='y')
mode = tf.placeholder(tf.bool, name='mode')
model = uabMakeNetwork_UNet.UnetModelCrop({'X':X, 'Y':y}, trainable=mode, input_size=input_size,
                                          batch_size=batch_size, start_filter_num=32)
# create graph
model.create_graph('X', class_num=2)

# evaluate on tiles
for rgb_name, gt_name in zip(rgb_list, gt_list):
    tile_dim = np.array(tile_size)
    # prepare the reader
    rgb = ersa_utils.load_file(rgb_name)
    gt = np.expand_dims(ersa_utils.load_file(gt_name), axis=-1)

    padding = np.array((model.get_overlap() / 2, model.get_overlap() / 2))
    block = np.dstack([rgb, gt]).astype(np.float32)
    block = uabUtilreader.pad_block(block, padding)
    tile_dim = tile_dim + padding * 2

    if model.config is None:
        model.config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=model.config) as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        model.load(model_dir, sess, epoch=75, best_model=False)

        patch_cnt = 0
        for patch in uabUtilreader.patchify(block, tile_dim, input_size, overlap=model.get_overlap()):
            rgb_patch = np.expand_dims(patch[:, :, :3] - img_mean, axis=0)
            gt_patch = patch[:, :, -1]

            pred = sess.run(model.output, feed_dict={model.inputs['X']: rgb_patch,
                                                     model.trainable: False})

            p = np.argmax(pred[0, :, :, :], axis=-1)
            g = patch[92:-92, 92:-92, -1]

            # measure fn objects
            lbl = measure.label(g)
            building_idx = np.unique(lbl)

            flag = False

            for cnt, idx in enumerate(building_idx):
                on_target = np.sum(p[np.where(lbl == idx)])
                building_size = np.sum(g[np.where(lbl == idx)])

                if building_size > 25:
                    if on_target/building_size <= 0.5:
                        flag = True
                        p[np.where(lbl == idx)] = 2
                        #break

                        g[np.where(lbl == idx)] = 2
                        gt_patch[92:-92, 92:-92] = g

            if flag:
                '''file_list = []
                for i in range(3):
                    ersa_utils.save_file(os.path.join(save_dir, '{}_rgb{}.jpg'.format(patch_cnt, i)), patch[:, :, i].
                                         astype(np.uint8))
                    file_list.append('{}_rgb{}.jpg'.format(patch_cnt, i))
                ersa_utils.save_file(os.path.join(save_dir, '{}_gt.png'.format(patch_cnt)), gt_patch.astype(np.uint8))
                file_list.append('{}_gt.png'.format(patch_cnt))
                f.write('{}\n'.format(' '.join(file_list)))
                patch_cnt += 1'''

                visualize_utils.compare_two_figure(patch[92:-92, 92:-92, :3].astype(np.uint8), gt_patch, show_fig=False)
                #visualize_utils.compare_two_figure(patch[92:-92, 92:-92, :3].astype(np.uint8), p, show_fig=False)
                plt.savefig(os.path.join(save_dir, '{}_gt.png'.format(patch_cnt)))
                patch_cnt += 1
                plt.close()

f.close()
