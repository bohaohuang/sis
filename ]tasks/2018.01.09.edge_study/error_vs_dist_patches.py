import os
import imageio
import tensorflow as tf
import numpy as np
import uabCrossValMaker
import uab_collectionFunctions
import util_functions
import uabUtilreader
from bohaoCustom import uabMakeNetwork_UNet

util_functions.tf_warn_level()

# settings
gpu = None
batch_size = 1
input_sizes = [572, 828, 1084, 1340, 1596, 1852, 2092, 2332, 2636]
tile_size = [5000, 5000]

for size in input_sizes:
    print('size: {}'.format(size))

    # make save dir
    img_save_dir = os.path.join(r'/hdd/Temp/patches', '{}'.format(size))
    if not os.path.exists(img_save_dir):
        os.makedirs(img_save_dir)

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

        # read the image
        img = []
        for c_dir, c_name in zip(parent_dir, file_name):
            img.append(imageio.imread(os.path.join(c_dir, c_name)))
        img = np.dstack(img).astype(np.float32) - img_mean

        # read the truth
        gt = np.expand_dims(imageio.imread(os.path.join(parent_dir_truth, file_name_truth)), axis=-1)

        img_gt = np.dstack([img, gt])

        # extract patches
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            model.load(model_dir, sess)

            cnt = 0
            for corner_h in np.floor(np.linspace(0, tile_size[0] - size, 14)).astype(np.int32):
                for corner_w in np.floor(np.linspace(0, tile_size[0] - size, 14)).astype(np.int32):
                    cnt += 1

                    block = uabUtilreader.crop_image(img_gt, [size, size], (corner_h, corner_w))
                    sub_img = block[:, :, :3]
                    sub_gt = block[:, :, -1]

                    pred = sess.run(model.output, feed_dict={model.inputs['X']: np.expand_dims(sub_img, axis=0),
                                                             model.trainable: False})
                    pred = np.argmax(pred[0, :, :, :], axis=2)
                    gt_cmp = sub_gt[92:size-92, 92:size-92]
                    error_map = np.abs(pred - gt_cmp)

                    save_name = '{}_{}.png'.format(tile_name, cnt)
                    imageio.imsave(os.path.join(img_save_dir, save_name), error_map)
