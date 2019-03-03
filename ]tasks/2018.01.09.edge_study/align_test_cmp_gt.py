import os
import imageio
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import sis_utils
import util_functions
import uab_collectionFunctions
from bohaoCustom import uabMakeNetwork_UNet


def evaluate_on_a_patch(chip_size,
                        X_batch,
                        pretrained_model_dir=
                        r'/hdd/Models/UnetCrop_inria_aug_grid_PS(572, 572)_BS5_EP100_LR0.0001_DS60_DR0.1_SFN32'):
    # experiment settings
    batch_size = 9  # mini-batch size
    learn_rate = 1e-4  # learning rate
    decay_step = 60  # learn rate dacay after 60 epochs
    decay_rate = 0.1  # learn rate decay to 0.1*before
    epochs = 100  # total number of epochs to rum
    start_filter_num = 32  # the number of filters at the first layer
    model_name = 'inria_aug_grid'  # a suffix for model name

    # make network
    # define place holder
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape=[None, chip_size[0], chip_size[1], 3], name='X')
    y = tf.placeholder(tf.int32, shape=[None, chip_size[0], chip_size[1], 1], name='y')
    mode = tf.placeholder(tf.bool, name='mode')
    model = uabMakeNetwork_UNet.UnetModelCrop({'X': X, 'Y': y},
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

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        model.load(pretrained_model_dir, sess)
        pred = sess.run(model.output, feed_dict={model.inputs['X']: X_batch,
                                                 model.trainable: False})
        pred = pred[0, :, :, :]
        pred = np.argmax(pred, axis=2)

    return pred


# get img mean
# prepare data
blCol = uab_collectionFunctions.uabCollection('inria')
img_mean = blCol.getChannelMeans([0, 1, 2])
img_dir, task_dir = sis_utils.get_task_img_folder()
large_ious = []
small_ious = []

# prepare tile
for city in ['austin', 'chicago', 'kitsap', 'tyrol-w', 'vienna']:
    for city_id in range(5):
        tile_path = r'/media/ei-edl01/data/uab_datasets/inria/data/Original_Tiles'
        tile_name = '{}{}_RGB.tif'.format(city, city_id + 1)
        gt_name = '{}{}_GT.tif'.format(city, city_id + 1)
        tile = imageio.imread(os.path.join(tile_path, tile_name)).astype(np.float32)
        gt = imageio.imread(os.path.join(tile_path, gt_name))
        tile -= img_mean

        util_functions.tf_warn_level()

        # test large image
        chip_size_large = 844
        X_batch = np.expand_dims(tile[:chip_size_large, :chip_size_large, :], axis=0)
        pred = evaluate_on_a_patch([chip_size_large, chip_size_large], X_batch)

        # test small image
        chip_size = 508
        X_batch_1 = np.expand_dims(tile[6:6 + chip_size, 6:6 + chip_size, :], axis=0)
        pred_1 = evaluate_on_a_patch([chip_size, chip_size], X_batch_1)
        X_batch_2 = np.expand_dims(tile[330:330 + chip_size, 6:6 + chip_size, :], axis=0)
        pred_2 = evaluate_on_a_patch([chip_size, chip_size], X_batch_2)
        X_batch_3 = np.expand_dims(tile[6:6 + chip_size, 330:330 + chip_size, :], axis=0)
        pred_3 = evaluate_on_a_patch([chip_size, chip_size], X_batch_3)
        X_batch_4 = np.expand_dims(tile[330:330 + chip_size, 330:330 + chip_size, :], axis=0)
        pred_4 = evaluate_on_a_patch([chip_size, chip_size], X_batch_4)

        # stitch
        pred_stitch = np.zeros_like(pred)
        pred_stitch[6:6 + 324, 6:6 + 324] = pred_1
        pred_stitch[330:330 + 324, 6:6 + 324] = pred_2
        pred_stitch[6:6 + 324, 330:330 + 324] = pred_3
        pred_stitch[330:330 + 324, 330:330 + 324] = pred_4

        # diff
        pred_diff = pred - pred_stitch
        gt_window = gt[92:chip_size_large - 92, 92:chip_size_large - 92]
        gt_plot = tile[92:chip_size_large-92,92:chip_size_large-92,:] + img_mean
        gt_cmp_large = 255 + np.zeros((gt_window.shape[0], gt_window.shape[1], 3), dtype=np.float32)
        gt_cmp_small = 255 + np.zeros((gt_window.shape[0], gt_window.shape[1], 3), dtype=np.float32)
        gt_plot = util_functions.add_mask(gt_plot, gt_window, [0, 0, 0], 255)
        gt_cmp_large = util_functions.add_mask(gt_cmp_large, gt_window, [0, 0, 0], 255)
        gt_cmp_large = util_functions.add_mask(gt_cmp_large, gt_window/255-pred, [255, 0, 0], 1)
        gt_cmp_large = util_functions.add_mask(gt_cmp_large, gt_window/255-pred, [0, 255, 0], -1)
        gt_cmp_small = util_functions.add_mask(gt_cmp_small, gt_window, [0, 0, 0], 255)
        gt_cmp_small = util_functions.add_mask(gt_cmp_small, gt_window/255-pred_stitch, [255, 0, 0], 1)
        gt_cmp_small = util_functions.add_mask(gt_cmp_small, gt_window/255-pred_stitch, [0, 255, 0], -1)

        # compute iou
        iou_pred = util_functions.iou_metric(gt_window[6:-6, 6:-6]/255, pred[6:-6, 6:-6]) * 100
        iou_stitch = util_functions.iou_metric(gt_window[6:-6, 6:-6]/255, pred_stitch[6:-6, 6:-6]) * 100
        if not (np.isnan(iou_pred) or np.isnan(iou_stitch)):
            large_ious.append(iou_pred)
            small_ious.append(iou_stitch)

        # show
        plt.figure(figsize=(13, 5))
        plt.subplot(131)
        plt.xticks([], [])
        plt.yticks([], [])
        plt.imshow(gt_plot/255)

        plt.subplot(132)
        plt.xticks([], [])
        plt.yticks([], [])
        plt.imshow(gt_cmp_large/255)
        plt.title('{:.3f}'.format(iou_pred))

        plt.subplot(133)
        plt.xticks([], [])
        plt.yticks([], [])
        plt.imshow(gt_cmp_small/255)
        plt.plot([0, 660], [6, 6], 'r--', linewidth=2)
        plt.plot([0, 660], [6 + 324, 6 + 324], 'r--', linewidth=2)
        plt.plot([0, 660], [6 + 324 * 2, 6 + 324 * 2], 'r--', linewidth=2)
        plt.plot([6, 6], [0, 660], 'r--', linewidth=2)
        plt.plot([6 + 324, 6 + 324], [0, 660], 'r--', linewidth=2)
        plt.plot([6 + 324 * 2, 6 + 324 * 2], [0, 660], 'r--', linewidth=2)
        plt.xlim(0, 660)
        plt.ylim(660, 0)
        plt.title('{:.3f}'.format(iou_stitch))

        plt.tight_layout()
        file_name = 'gt_cmp' + tile_name.split('_')[0]
        plt.savefig(os.path.join(img_dir, file_name))
        #plt.show()

np.save(os.path.join(task_dir, 'iou_records.npy'), [large_ious, small_ious])
