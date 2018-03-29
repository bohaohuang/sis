import os
import imageio
import tensorflow as tf
import numpy as np
import uabDataReader
import uabCrossValMaker
import uabUtilreader
import util_functions
import uab_collectionFunctions
from bohaoCustom import uabMakeNetwork_DeepLabV2

# settings
gpu = 0
util_functions.tf_warn_level()
batch_size = 1
input_sizes = [321]
for size in input_sizes:
    print('Evaluating at size {} ...'.format(size))
    input_size = [size, size]
    tile_size = [5000, 5000]
    save_dir = '/hdd/Temp/IGARSS2018/Deeplab{}'.format(size)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    model_dir = r'/hdd6/Models/DeepLab_rand_grid/DeeplabV3_res101_inria_aug_grid_1_PS(321, 321)_BS5_EP100_LR1e-05_DS40_DR0.1_SFN32'

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

    # evaluate on tiles
    for file_name, file_name_truth in zip(file_list_valid, file_list_valid_truth):
        tile_name = file_name_truth.split('_')[0]
        print('Evaluating {} ...'.format(tile_name))

        # make the model
        # define place holder
        X = tf.placeholder(tf.float32, shape=[None, input_size[0], input_size[1], 3], name='X')
        y = tf.placeholder(tf.int32, shape=[None, input_size[0], input_size[1], 1], name='y')
        mode = tf.placeholder(tf.bool, name='mode')
        model = uabMakeNetwork_DeepLabV2.DeeplabV3({'X': X, 'Y': y},
                                                   trainable=mode,
                                                   input_size=input_size,
                                                   batch_size=5,
                                                   start_filter_num=32)
        # create graph
        model.create_graph('X', class_num=2)

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
                                                padding=np.array((model.get_overlap()/2, model.get_overlap()/2)),
                                                isTrain=False)
        rManager = reader.readManager

        # run the model
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            model.load(model_dir, sess)
            model.model_name = model_dir.split('/')[-1]
            result = model.test('X', sess, rManager)

        # patchify gt
        gt_img = np.expand_dims(imageio.imread(os.path.join(parent_dir_truth, file_name_truth)), axis=-1)
        gt_img = uabUtilreader.pad_block(gt_img, np.array((model.get_overlap()/2, model.get_overlap()/2)))
        gtReader = uabUtilreader.patchify(gt_img, tile_size+np.array((model.get_overlap(), model.get_overlap())),
                                          input_size, overlap=model.get_overlap())

        patch_num = result.shape[0]
        for n in range(patch_num):
            pred = np.argmax(result[n, :, :, :], axis=2).astype(np.uint8)
            pred_gt = next(gtReader)[:,:,0]*255

            try:
                imageio.imsave(os.path.join(save_dir, '{}_pred_{}.png'.format(tile_name, n)), pred)
                imageio.imsave(os.path.join(save_dir, '{}_gt_{}.png').format(tile_name, n), pred_gt)
            except ValueError as e:
                print(e)
                continue

        tf.reset_default_graph()
