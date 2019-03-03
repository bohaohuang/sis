import os
import time
import numpy as np
import tensorflow as tf
import uabCrossValMaker
import uab_collectionFunctions
import sis_utils
import util_functions
from bohaoCustom import uabMakeNetwork_DeepLabV2

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
    for size in [321, 520, 736, 832, 1088, 1344, 1600, 1856, 2096, 2640]:
        input_size = [size, size]

        tf.reset_default_graph()

        model_dir = r'/hdd6/Models/DeepLab_rand_grid/DeeplabV3_res101_inria_aug_grid_0_PS(321, 321)_BS5_EP100_LR1e-05_DS40_DR0.1_SFN32'
        blCol = uab_collectionFunctions.uabCollection('inria')
        blCol.readMetadata()
        file_list, parent_dir = blCol.getAllTileByDirAndExt([0, 1, 2])
        file_list_truth, parent_dir_truth = blCol.getAllTileByDirAndExt(4)
        idx, file_list = uabCrossValMaker.uabUtilGetFolds(None, file_list, 'force_tile')
        idx_truth, file_list_truth = uabCrossValMaker.uabUtilGetFolds(None, file_list_truth, 'force_tile')
        # use first 5 tiles for validation
        file_list_valid = uabCrossValMaker.make_file_list_by_key(
            idx, file_list, [i for i in range(0, 6)],
            filter_list=['bellingham', 'bloomington', 'sfo', 'tyrol-e', 'innsbruck'])
        file_list_valid_truth = uabCrossValMaker.make_file_list_by_key(
            idx_truth, file_list_truth, [i for i in range(0, 6)],
            filter_list=['bellingham', 'bloomington', 'sfo', 'tyrol-e', 'innsbruck'])
        img_mean = blCol.getChannelMeans([0, 1, 2])

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
        start_time = time.time()
        iou = model.evaluate(file_list_valid, file_list_valid_truth, parent_dir, parent_dir_truth,
                             input_size, tile_size, batch_size, img_mean, model_dir, gpu,
                             save_result=False)
        duration = time.time() - start_time

        file_name = '{}_{}_{}.npy'.format(model.model_name, size, run_repeat)
        np.save(os.path.join(save_dir, file_name), [iou, duration])
