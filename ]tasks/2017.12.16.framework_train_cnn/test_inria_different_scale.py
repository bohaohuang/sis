import os
import time
import tensorflow as tf
import numpy as np
import uabCrossValMaker
import uab_collectionFunctions
import utils
from bohaoCustom import uabMakeNetwork_UNet

# settings
gpu = 1
batch_size = 1
#input_sizes = [828, 1084, 1340, 1596, 572, 1852, 2108, 2364, 2620]
#input_sizes = [540, 572, 620, 684, 796]
#batch_sizes = [6, 5, 4, 3, 2]
input_sizes = [412, 460, 476, 492, 508]
batch_sizes = [15, 10, 9, 8, 7]
tile_size = [5000, 5000]
img_dir, task_dir = utils.get_task_img_folder()

for cnt, size in enumerate(input_sizes):
    start_time = time.time()

    tf.reset_default_graph()
    input_size = [size, size]

    model_dir = r'/hdd/Models/exp2/UnetCrop_inria_aug_grid_1_PS({}, {})_BS{}_EP100_LR0.0001_DS60_DR0.1_SFN32'.\
        format(size, size, batch_sizes[cnt])
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
    X = tf.placeholder(tf.float32, shape=[None, 1052, 1052, 3], name='X')
    y = tf.placeholder(tf.int32, shape=[None, 1052, 1052, 1], name='y')
    mode = tf.placeholder(tf.bool, name='mode')
    model = uabMakeNetwork_UNet.UnetModelCrop({'X':X, 'Y':y},
                                              trainable=mode,
                                              input_size=[1052, 1052],
                                              batch_size=1)
    # create graph
    model.create_graph('X', class_num=2)

    # evaluate on tiles
    iou_return = model.evaluate(file_list_valid, file_list_valid_truth, parent_dir, parent_dir_truth,
                                [1052, 1052], tile_size, batch_size, img_mean, model_dir, gpu, save_result=False)
    duration = time.time() - start_time

    iou_return['time']  = duration
    np.save(os.path.join(task_dir, '{}_samesize.npy'.format(size)), iou_return)
