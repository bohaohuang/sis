import os
import time
import imageio
import tensorflow as tf
import numpy as np
import uabDataReader
import uabCrossValMaker
import uab_collectionFunctions
import utils
import util_functions
from bohaoCustom import uabMakeNetwork_UNet

util_functions.tf_warn_level()

# settings
gpu = None
batch_size = 1
input_sizes = [572, 828, 1084, 1340, 1596, 1852, 2092, 2332, 2636]
tile_size = [5000, 5000]
img_dir, task_dir = utils.get_task_img_folder()
img_save_dir = os.path.join(img_dir, 'dist_error')
if not os.path.exists(img_save_dir):
    os.makedirs(img_save_dir)

for cnt, size in enumerate(input_sizes):
    start_time = time.time()

    tf.reset_default_graph()
    input_size = [size, size]

    model_dir = r'/hdd/Models/UnetCrop_inria_aug_grid_PS(572, 572)_BS5_EP100_LR0.0001_DS60_DR0.1_SFN32'
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
    X = tf.placeholder(tf.float32, shape=[None, size, size, 3], name='X')
    y = tf.placeholder(tf.int32, shape=[None, size, size, 1], name='y')
    mode = tf.placeholder(tf.bool, name='mode')
    model = uabMakeNetwork_UNet.UnetModelCrop({'X':X, 'Y':y},
                                              trainable=mode,
                                              input_size=input_size,
                                              batch_size=1)
    # create graph
    model.create_graph('X', class_num=2)

    # evaluate on tiles
    for file_name, file_name_truth in zip(file_list_valid, file_list_valid_truth):
        tile_name = file_name_truth.split('_')[0]
        print('Evaluating {} ... '.format(tile_name))
        # prepare the reader
        reader = uabDataReader.ImageLabelReader(gtInds=[0],
                                                dataInds=[0],
                                                nChannels=3,
                                                parentDir=parent_dir,
                                                chipFiles=[file_name],
                                                chip_size=input_size,
                                                tile_size=tile_size,
                                                batchSize=batch_size,
                                                block_mean=img_mean,
                                                overlap=model.get_overlap(),
                                                padding=np.array((model.get_overlap() / 2, model.get_overlap() / 2)),
                                                isTrain=False)
        rManager = reader.readManager

        # run the model
        pred = model.run(pretrained_model_dir=model_dir,
                         test_reader=rManager,
                         tile_size=tile_size,
                         patch_size=input_size,
                         gpu=gpu)
        truth_label_img = imageio.imread(os.path.join(parent_dir_truth, file_name_truth))

        error_img = np.abs(pred-truth_label_img)

        img_name = '{}_{}.png'.format(size, tile_name)
        imageio.imsave(os.path.join(img_save_dir, img_name), error_img)
