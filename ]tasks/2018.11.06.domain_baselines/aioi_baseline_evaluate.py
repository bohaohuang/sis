import tensorflow as tf
import uab_collectionFunctions
from nn import nn_utils
from bohaoCustom import uabMakeNetwork_UNet


if __name__ == '__main__':
    # settings
    nn_utils.tf_warn_level(3)
    model_dir = r'/hdd6/Models/UNET_rand_gird/UnetCrop_inria_aug_grid_0_PS(572, 572)_BS5_' \
                r'EP100_LR0.0001_DS60_DR0.1_SFN32'
    gpu = 0
    batch_size = 1
    input_size = [572, 572]
    for city_name in ['Arlington', 'Atlanta', 'Austin', 'DC', 'NewHaven', 'NewYork', 'SanFrancisco', 'Seekonk']:
        tf.reset_default_graph()

        blCol = uab_collectionFunctions.uabCollection(city_name)
        blCol.readMetadata()
        file_list, parent_dir = blCol.getAllTileByDirAndExt([0, 1, 2])
        file_list_truth, parent_dir_truth = blCol.getAllTileByDirAndExt(3)
        img_mean = blCol.getChannelMeans([0, 1, 2])

        # make the model
        # define place holder
        X = tf.placeholder(tf.float32, shape=[None, input_size[0], input_size[1], 3], name='X')
        Z = tf.placeholder(tf.float32, shape=[None, input_size[0], input_size[1], 3], name='Z')
        y = tf.placeholder(tf.int32, shape=[None, input_size[0], input_size[1], 1], name='y')
        mode = tf.placeholder(tf.bool, name='mode')
        model = uabMakeNetwork_UNet.UnetModelCrop({'X': X, 'Y': y}, trainable=mode, input_size=input_size,
                                                  batch_size=batch_size)
        # create graph
        model.create_graph('X', class_num=2)

        # evaluate on tiles
        model.evaluate(file_list, file_list_truth, parent_dir, parent_dir_truth,
                       input_size, None, batch_size, img_mean, model_dir, gpu,
                       save_result_parent_dir='domain_baseline/base', ds_name=city_name, best_model=False,
                       load_epoch_num=95, show_figure=False)
