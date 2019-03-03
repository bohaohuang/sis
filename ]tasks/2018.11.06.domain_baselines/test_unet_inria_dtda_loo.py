import os
import tensorflow as tf
import sis_utils
import ersa_utils
import uabCrossValMaker
import uab_collectionFunctions
from nn import nn_utils
from bohaoCustom import uabMakeNetwork_UNet


if __name__ == '__main__':
    # settings
    gpu = 0
    batch_size = 1
    input_size = [572, 572]
    tile_size = [5000, 5000]
    city_list = ['austin', 'chicago', 'kitsap', 'tyrol-w', 'vienna']
    nn_utils.tf_warn_level(3)

    img_dir, task_dir = sis_utils.get_task_img_folder()

    for city_id in [0]:
        path_to_save = os.path.join(task_dir, 'dtda', city_list[city_id], 'shift_dict.pkl')
        shift_dict = ersa_utils.load_file(path_to_save)

        model_dir = r'/hdd6/Models/domain_baseline/contorl_valid/UnetDTDA_inria_aug_leave_{}_0_iid_PS(572, 572)_BS8_' \
                    r'EP60_LR1e-06_DS40.0_DR0.1'.format(city_id)

        tf.reset_default_graph()

        blCol = uab_collectionFunctions.uabCollection('inria')
        blCol.readMetadata()
        file_list, parent_dir = blCol.getAllTileByDirAndExt([0, 1, 2])
        file_list_truth, parent_dir_truth = blCol.getAllTileByDirAndExt(4)
        idx, file_list = uabCrossValMaker.uabUtilGetFolds(None, file_list, 'force_tile')
        idx_truth, file_list_truth = uabCrossValMaker.uabUtilGetFolds(None, file_list_truth, 'force_tile')
        # use first 5 tiles for validation
        exclude_cities = [city_list[a] for a in range(5) if a != city_id]
        file_list_valid = uabCrossValMaker.make_file_list_by_key(
            idx, file_list, [i for i in range(0, 6)],
            filter_list=['bellingham', 'bloomington', 'sfo', 'tyrol-e', 'innsbruck'] + exclude_cities)
        file_list_valid_truth = uabCrossValMaker.make_file_list_by_key(
            idx_truth, file_list_truth, [i for i in range(0, 6)],
            filter_list=['bellingham', 'bloomington', 'sfo', 'tyrol-e', 'innsbruck']+ exclude_cities)
        img_mean = blCol.getChannelMeans([0, 1, 2])

        # make the model
        # define place holder
        X = tf.placeholder(tf.float32, shape=[None, input_size[0], input_size[1], 3], name='X')
        Z = tf.placeholder(tf.float32, shape=[None, input_size[0], input_size[1], 3], name='Z')
        y = tf.placeholder(tf.int32, shape=[None, input_size[0], input_size[1], 1], name='y')
        mode = tf.placeholder(tf.bool, name='mode')
        model = uabMakeNetwork_UNet.UnetModelDTDA({'X': X, 'Z': Z, 'Y': y}, trainable=mode, input_size=input_size, batch_size=5)
        # create graph
        model.create_graph('X', 'Z', class_num=2)

        # evaluate on tiles
        model.evaluate(file_list_valid, file_list_valid_truth, parent_dir, parent_dir_truth,
                       input_size, tile_size, batch_size, img_mean, model_dir, gpu,
                       save_result_parent_dir='domain_baseline2', ds_name='inria', best_model=False,
                       load_epoch_num=55, show_figure=False)
