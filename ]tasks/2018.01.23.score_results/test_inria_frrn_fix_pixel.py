import tensorflow as tf
import uabCrossValMaker
import uab_collectionFunctions
import util_functions
from bohaoCustom import uabMakeNetwork_FRRN

# settings
gpu = None
batch_sizes = [10,  8,   7,   6,   5,   4,   3,   2,   1]
patch_sizes = [160, 176, 192, 208, 224, 256, 288, 352, 496]
tile_size = [5000, 5000]
util_functions.tf_warn_level(3)

for patch_size, batch_size in zip(patch_sizes, batch_sizes):
    input_size = [patch_size, patch_size]
    for runId in range(5):
        tf.reset_default_graph()

        model_dir = r'/hdd/Models/FRRN_fix_pixel/FRRN_inria_aug_psbs_{}_PS({}, {})_BS{}_EP100_LR0.0001_DS60_DR0.1_SFN32'.\
            format(runId, patch_size, patch_size, batch_size)
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
        model = uabMakeNetwork_FRRN.FRRN({'X':X, 'Y':y},
                                         trainable=mode,
                                         input_size=input_size,
                                         batch_size=5, start_filter_num=32)
        # create graph
        model.create_graph('X', class_num=2)

        # evaluate on tiles
        model.evaluate(file_list_valid, file_list_valid_truth, parent_dir, parent_dir_truth,
                       input_size, tile_size, batch_size, img_mean, model_dir, gpu,
                       save_result_parent_dir='fix_pixel', ds_name='inria')
