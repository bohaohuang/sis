import os
import time
import numpy as np
import tensorflow as tf
import uabCrossValMaker
import uab_collectionFunctions
import sis_utils
import util_functions
from bohaoCustom import uabMakeNetwork_UNet

# settings
gpu = None
batch_size = 1
tile_size = [5000, 5000]
util_functions.tf_warn_level(3)
img_dir, task_dir = sis_utils.get_task_img_folder()
save_dir = os.path.join(task_dir, 'train_patch')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

for run_repeat in range(1):
    for size in [572, 828, 1084, 1340, 1596, 1852, 2092, 2332, 2636]:
        input_size = [size, size]

        tf.reset_default_graph()

        model_dir = r'/hdd6/Models/UNET_rand_gird/UnetCrop_spca_aug_grid_0_PS(572, 572)_BS5_EP100_LR0.0001_DS60_DR0.1_SFN32'
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
        start_time = time.time()
        iou = model.evaluate(file_list_valid, file_list_valid_truth, parent_dir, parent_dir_truth,
                             input_size, tile_size, batch_size, img_mean, model_dir, gpu,
                             save_result=False)
        duration = time.time() - start_time

        file_name = '{}_{}_{}_spca.npy'.format(model.model_name, size, run_repeat)
        np.save(os.path.join(save_dir, file_name), [iou, duration])
