import os
import tensorflow as tf
import ersa_utils
import uabCrossValMaker
import uab_collectionFunctions
from nn import nn_utils
from bohaoCustom import uabMakeNetwork_UNet


def get_pretrained_weights(weight_dir, model_dir):
    save_name = os.path.join(weight_dir, 'weight.pkl')

    if not os.path.exists(save_name):
        X = tf.placeholder(tf.float32, shape=[None, input_size[0], input_size[1], 3], name='X')
        y = tf.placeholder(tf.int32, shape=[None, input_size[0], input_size[1], 1], name='y')
        mode = tf.placeholder(tf.bool, name='mode')
        model = uabMakeNetwork_UNet.UnetModelCrop({'X': X, 'Y': y},
                                                  trainable=mode,
                                                  input_size=input_size,
                                                  start_filter_num=32)
        model.create_graph('X', class_num=2)
        train_vars = [v for v in tf.trainable_variables()]

        weight_dict = dict()

        with tf.Session() as sess:
            model.load(model_dir, sess, epoch=95)
            for v in train_vars:
                theta = sess.run(v)
                weight_dict[v.name] = theta
        ersa_utils.save_file(save_name, weight_dict)
    else:
        weight_dict = ersa_utils.load_file(save_name)

    tf.reset_default_graph()
    return weight_dict


# settings
gpu = 1
nn_utils.tf_warn_level(3)
batch_size = 5
input_size = [572, 572]
tile_size = [5000, 5000]
city_list = ['austin', 'chicago', 'kitsap', 'tyrol-w', 'vienna']
pre_model_dir = r'/hdd6//Models/Inria_Domain_LOO/UnetCrop_inria_aug_leave_{}_0_PS(572, 572)_BS5_' \
            r'EP100_LR0.0001_DS60_DR0.1_SFN32'
weight_dir = r'/media/ei-edl01/user/bh163/tasks/2018.11.06.domain_baselines/dtda/{}'



for city_id in [0]:
    pre_model_dir = pre_model_dir.format(city_id)
    weight_dir = weight_dir.format(city_list[city_id])
    weight_dict = get_pretrained_weights(weight_dir, pre_model_dir)

    model_dir = r'/hdd6/Models/domain_baseline/UnetDTDA_inria_aug_leave_{}_0_PS(572, 572)_BS8_' \
                r'EP100_LR1e-05_DS60_DR0.1'.format(city_id)
    exclude_city_list = [city_list[i] for i in range(5) if i != city_id]

    tf.reset_default_graph()

    blCol = uab_collectionFunctions.uabCollection('inria')
    blCol.readMetadata()
    file_list, parent_dir = blCol.getAllTileByDirAndExt([0, 1, 2])
    file_list_truth, parent_dir_truth = blCol.getAllTileByDirAndExt(4)
    idx, file_list = uabCrossValMaker.uabUtilGetFolds(None, file_list, 'force_tile')
    idx_truth, file_list_truth = uabCrossValMaker.uabUtilGetFolds(None, file_list_truth, 'force_tile')
    # use first 5 tiles for validation
    file_list_valid = uabCrossValMaker.make_file_list_by_key(
        idx, file_list, [i for i in range(0, 6)],
        filter_list=['bellingham', 'bloomington', 'sfo', 'tyrol-e', 'innsbruck'] + exclude_city_list)
    file_list_valid_truth = uabCrossValMaker.make_file_list_by_key(
        idx_truth, file_list_truth, [i for i in range(0, 6)],
        filter_list=['bellingham', 'bloomington', 'sfo', 'tyrol-e', 'innsbruck'] + exclude_city_list)
    img_mean = blCol.getChannelMeans([0, 1, 2])

    # make the model
    # define place holder
    X = tf.placeholder(tf.float32, shape=[None, input_size[0], input_size[1], 3], name='X')
    Z = tf.placeholder(tf.float32, shape=[None, input_size[0], input_size[1], 3], name='Z')
    y = tf.placeholder(tf.int32, shape=[None, input_size[0], input_size[1], 1], name='y')
    mode = tf.placeholder(tf.bool, name='mode')
    model = uabMakeNetwork_UNet.UnetModelDTDA({'X': X, 'Z': Z, 'Y': y},
                                              trainable=mode,
                                              input_size=input_size,
                                              batch_size=5)
    # create graph
    model.create_graph('X', 'Y', class_num=2)

    # evaluate on tiles
    model.load_source_weights(pre_model_dir)
    model.evaluate(file_list_valid, file_list_valid_truth, parent_dir, parent_dir_truth,
                   input_size, tile_size, batch_size, img_mean, model_dir, gpu,
                   save_result_parent_dir='domain_baseline', ds_name='inria',
                   best_model=False, show_figure=False)
