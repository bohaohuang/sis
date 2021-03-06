import os
import time
import tensorflow as tf
import numpy as np
import uabCrossValMaker
import uab_collectionFunctions
import sis_utils
from bohaoCustom import uabMakeNetwork_ResFCN

# settings
gpu = 0
batch_size = 1
#input_sizes = [224, 480, 736, 992, 1248, 1504, 1760, 2016, 2272, 2528]
input_sizes = [1504, 1760, 2016, 2272, 2528]
tile_size = [5000, 5000]
img_dir, task_dir = sis_utils.get_task_img_folder()

for size in input_sizes:
    start_time = time.time()

    tf.reset_default_graph()
    input_size = [size, size]

    model_dir = r'/hdd/Models/ResFcn_PS(224, 224)_BS5_EP100_LR0.0001_DS60_DR0.1_SFN32'
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
    model = uabMakeNetwork_ResFCN.ResFcnModel({'X':X, 'Y':y},
                                              trainable=mode,
                                              input_size=input_size,
                                              batch_size=5,
                                              start_filter_num=32)
    # create graph
    model.create_graph('X', class_num=2)

    # evaluate on tiles
    iou_return = model.evaluate(file_list_valid, file_list_valid_truth, parent_dir, parent_dir_truth,
                                input_size, tile_size, batch_size, img_mean, model_dir, gpu, save_result=False)
    duration = time.time() - start_time

    iou_return['time']  = duration
    np.save(os.path.join(task_dir, '{}_resfcn.npy'.format(size)), iou_return)
