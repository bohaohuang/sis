import os
import time
import tensorflow as tf
import numpy as np
import uabDataReader
import uabCrossValMaker
import uabUtilreader
import uab_collectionFunctions
from bohaoCustom import uabMakeNetwork_UNet

# settings
gpu = 0
batch_size = 1
input_sizes = [224, 480, 736, 992, 1248, 1504, 1760, 2016, 2272, 2528]
tile_size = [5000, 5000]
save_dir = '/hdd/Temp/IGARSS2018/ResFcn'
model_dir = r'/hdd/Models/Unet_inria_aug_grid_0_PS(224, 224)_BS10_EP100_LR0.0001_DS60_DR0.1_SFN32'

save_file_name = os.path.join(save_dir, 'run_time.txt')
with open(save_file_name, 'w') as f:
    pass

for size in input_sizes:
    input_size = [size, size]

    # prepare data
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

    start_time = time.time()
    # evaluate on tiles
    for file_name, file_name_truth in zip(file_list_valid, file_list_valid_truth):
        tile_name = file_name_truth.split('_')[0]
        print('Evaluating {} ...'.format(tile_name))

        # make the model
        # define place holder
        X = tf.placeholder(tf.float32, shape=[None, input_size[0], input_size[1], 3], name='X')
        y = tf.placeholder(tf.int32, shape=[None, input_size[0], input_size[1], 1], name='y')
        mode = tf.placeholder(tf.bool, name='mode')
        model = uabMakeNetwork_UNet.UnetModel({'X': X, 'Y': y},
                                              trainable=mode,
                                              input_size=input_size,
                                              batch_size=5,
                                              start_filter_num=32)
        # create graph
        model.create_graph('X', class_num=2)
        pad = np.array((model.get_overlap()/2, model.get_overlap()/2))

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
                                                padding=pad,
                                                isTrain=False)
        rManager = reader.readManager

        # run the model
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            model.load(model_dir, sess)
            model.model_name = model_dir.split('/')[-1]
            result = model.test('X', sess, rManager)
        image_pred = uabUtilreader.un_patchify(result, tile_size, input_size)
        tf.reset_default_graph()

    run_time = time.time() - start_time

    # save run time
    print(run_time)
    with open(save_file_name, 'a') as f:
        f.write('{} {}\n'.format(size, run_time))
