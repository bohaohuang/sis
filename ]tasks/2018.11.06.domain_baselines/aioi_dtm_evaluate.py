import tensorflow as tf
import uab_collectionFunctions
from nn import nn_utils
from bohaoCustom import uabMakeNetwork_UNet


if __name__ == '__main__':
    # settings
    nn_utils.tf_warn_level(3)
    gpu = 1
    batch_size = 1
    input_size = [2044, 2044]
    for city_name in ['Arlington', 'Austin', 'DC', 'NewHaven', 'NewYork', 'Norfolk', 'SanFrancisco', 'Seekonk']:
    #for city_name in ['Arlington', 'Atlanta', 'Austin', 'DC', 'NewHaven', 'NewYork', 'Norfolk', 'SanFrancisco', 'Seekonk']:
        #model_dir = r'/hdd6/Models/domain_baseline/dtm/UnetDTDA_inria_{}_0_iid_PS(572, 572)_BS8_' \
        #            r'EP60_LR1e-06_DS80_DR0.1'.format(city_name)
        #model_dir = r'/hdd6/Models/domain_baseline/distance/UnetCrop_inria_distance_xregion_5050_{}_1_PS(572, 572)_BS5_' \
        #            r'EP40_LR1e-05_DS30_DR0.1_SFN32'.format(city_name)
        model_dir = r'/hdd6/Models/Inria_Domain_Selection/UnetCrop_eval_aioi_mmd_xregion_5050_{}_0_PS(572, 572)_BS5_' \
                    r'EP40_LR5e-05_DS30_DR0.1_SFN32'.format(city_name)

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

        if 'DTDA' in model_dir:
            model = uabMakeNetwork_UNet.UnetModelDTDA({'X': X, 'Z': Z, 'Y': y}, trainable=mode, input_size=input_size,
                                                      batch_size=batch_size)
            # create graph
            model.create_graph('X', 'Z', class_num=2)

            # evaluate on tiles
            model.evaluate(file_list, file_list_truth, parent_dir, parent_dir_truth,
                           input_size, None, batch_size, img_mean, model_dir, gpu,
                           save_result_parent_dir='domain_baseline/dtm', ds_name=city_name, best_model=False,
                           show_figure=False)
        else:
            model = uabMakeNetwork_UNet.UnetModelCrop({'X': X, 'Y': y}, trainable=mode, input_size=input_size,
                                                      batch_size=batch_size)
            # create graph
            model.create_graph('X', class_num=2)

            # evaluate on tiles
            model.evaluate(file_list, file_list_truth, parent_dir, parent_dir_truth,
                           input_size, None, batch_size, img_mean, model_dir, gpu,
                           save_result_parent_dir='domain_baseline/mmd', ds_name=city_name, best_model=True,
                           show_figure=False)
