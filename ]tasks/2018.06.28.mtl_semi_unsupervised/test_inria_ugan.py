import tensorflow as tf
import uabCrossValMaker
import uab_collectionFunctions
import util_functions
from bohaoCustom import uabMakeNetwork_UNet

# settings
gpu = 0
batch_size = 5
input_size = [572, 572]
tile_size = [5000, 5000]
util_functions.tf_warn_level(3)
city_list = ['austin', 'chicago', 'kitsap', 'tyrol-w', 'vienna']

model_list = [#r'/hdd6/Models/Inria_GAN/new/UnetGAN_inria_gan_0_2_PS(572, 572)_BS4_EP30_LR0.0001_1e-06_1e-06_DS30.0_30.0_20.0_DR0.1_0.1_0.1',
              #r'/hdd6/Models/Inria_GAN/new/UnetGAN_inria_gan_0_2_PS(572, 572)_BS4_EP30_LR0.0001_1e-06_1e-06_DS20.0_30.0_20.0_DR0.1_0.1_0.1',
              #r'/hdd6/Models/Inria_GAN/new/UnetGAN_inria_gan_0_2_PS(572, 572)_BS4_EP30_LR0.0001_1e-05_1e-06_DS30.0_10.0_20.0_DR0.1_0.1_0.1',
              #r'/hdd6/Models/Inria_GAN/new/UnetGAN_inria_gan_0_2_PS(572, 572)_BS4_EP30_LR0.0001_1e-05_1e-06_DS30.0_10.0_30.0_DR0.1_0.1_0.1',
              #r'/hdd6/Models/Inria_GAN/new/UnetGAN_inria_gan_0_2_PS(572, 572)_BS4_EP30_LR0.0001_1e-06_1e-06_DS20.0_20.0_20.0_DR0.1_0.1_0.1',
              #r'/hdd6/Models/Inria_GAN/new/UnetGAN_inria_gan_0_2_PS(572, 572)_BS4_EP30_LR0.0001_1e-05_1e-06_DS20.0_20.0_20.0_DR0.1_0.1_0.1',
              #r'/hdd6/Models/Inria_GAN/new/UnetGAN_inria_gan_0_2_PS(572, 572)_BS4_EP30_LR1e-05_1e-05_1e-06_DS20.0_20.0_20.0_DR0.1_0.1_0.1',
              #r'/hdd6/Models/Inria_GAN/new/UnetGAN_inria_gan_0_2_PS(572, 572)_BS4_EP30_LR1e-05_1e-06_1e-06_DS20.0_20.0_20.0_DR0.1_0.1_0.1',
              #r'/hdd6/Models/Inria_GAN/new/UnetGAN_inria_gan_0_2_PS(572, 572)_BS4_EP60_LR0.0001_1e-06_1e-06_DS30.0_30.0_30.0_DR0.1_0.1_0.1',
              #r'/hdd6/Models/Inria_GAN/new/UnetGAN_inria_gan_0_2_PS(572, 572)_BS4_EP60_LR0.0001_1e-06_1e-06_DS60.0_30.0_30.0_DR0.1_0.1_0.1',
              #r'/hdd6/Models/Inria_GAN/new/UnetGAN_inria_gan_0_2_PS(572, 572)_BS4_EP60_LR0.0001_1e-05_1e-06_DS30.0_60.0_30.0_DR0.1_0.1_0.1'
              #r'/hdd6/Models/Inria_GAN/new/UnetGAN_inria_gan_real_0_2_PS(572, 572)_BS4_EP30_LR0.0001_1e-06_1e-06_DS30.0_20.0_30.0_DR0.1_0.1_0.1',
              #r'/hdd6/Models/Inria_GAN/new/UnetGAN_inria_gan_real_0_2_PS(572, 572)_BS4_EP30_LR0.0001_1e-05_1e-06_DS30.0_10.0_30.0_DR0.1_0.1_0.1',
              #r'/hdd6/Models/Inria_GAN/new/UnetGAN_inria_gan_real_0_2_PS(572, 572)_BS4_EP30_LR0.0001_1e-06_1e-06_DS30.0_30.0_30.0_DR0.1_0.1_0.1',
              #r'/hdd6/Models/Inria_GAN/UnetGAN_inria_gan_real_0_2_PS(572, 572)_BS4_EP50_LR0.0001_1e-05_1e-06_DS30.0_10.0_30.0_DR0.1_0.1_0.1',
              #r'/hdd6/Models/Inria_GAN/UnetGAN_inria_gan_real_1_2_PS(572, 572)_BS4_EP50_LR0.0001_1e-06_1e-06_DS30.0_20.0_30.0_DR0.1_0.1_0.1',
              #r'/hdd6/Models/Inria_GAN/UnetGAN_inria_gan_real_1_2_PS(572, 572)_BS4_EP50_LR0.0001_1e-05_1e-06_DS30.0_20.0_30.0_DR0.1_0.1_0.1',
              #r'/hdd6/Models/Inria_GAN/UnetGAN_inria_gan_real_4_2_PS(572, 572)_BS4_EP50_LR0.0001_1e-05_1e-06_DS30.0_10.0_30.0_DR0.1_0.1_0.1',
              #r'/hdd6/Models/Inria_GAN/fixed/UnetGAN_inria_gan_real_0_2_fixed_PS(572, 572)_BS4_EP35_LR0.0001_1e-05_1e-06_DS30.0_10.0_30.0_DR0.1_0.1_0.1',
              #r'/hdd6/Models/Inria_GAN/fixed/UnetGAN_inria_gan_real_1_2_fixed_PS(572, 572)_BS4_EP35_LR0.0001_1e-05_1e-06_DS30.0_10.0_30.0_DR0.1_0.1_0.1',
              #r'/hdd6/Models/Inria_GAN/fixed/UnetGAN_inria_gan_real_3_2_fixed_PS(572, 572)_BS4_EP35_LR0.0001_1e-05_1e-06_DS30.0_10.0_30.0_DR0.1_0.1_0.1',
              #r'/hdd6/Models/Inria_GAN/fixed/UnetGAN_inria_gan_real_2_2_fixed_PS(572, 572)_BS4_EP35_LR0.0001_1e-05_1e-06_DS30.0_10.0_30.0_DR0.1_0.1_0.1',
              #r'/hdd6/Models/Inria_GAN/fixed/UnetGAN_inria_gan_real_0_2_PS(572, 572)_BS4_EP50_LR0.0001_1e-05_1e-06_DS30.0_10.0_30.0_DR0.1_0.1_0.1',
              #r'/hdd6/Models/Inria_GAN/fixed/UnetGAN_inria_gan_real_1_2_PS(572, 572)_BS4_EP50_LR0.0001_1e-05_1e-06_DS30.0_10.0_30.0_DR0.1_0.1_0.1',
              #r'/hdd6/Models/Inria_GAN/fixed/UnetGAN_inria_gan_real_2_2_PS(572, 572)_BS4_EP50_LR0.0001_1e-05_1e-06_DS30.0_10.0_30.0_DR0.1_0.1_0.1',
              #r'/hdd6/Models/Inria_GAN/fixed/UnetGAN_inria_gan_real_3_2_PS(572, 572)_BS4_EP50_LR0.0001_1e-05_1e-06_DS30.0_10.0_30.0_DR0.1_0.1_0.1',
              #r'/hdd6/Models/Inria_GAN/UnetGAN_inria_gan_control_0_2_PS(572, 572)_BS4_EP30_LR0.0001_1e-05_1e-06_DS30.0_10.0_30.0_DR0.1_0.1_0.1',
              #r'/hdd6/Models/Inria_GAN/UnetGAN_inria_gan_0821_1_2_PS(572, 572)_BS4_EP50_LR0.0001_1e-05_1e-06_DS30.0_10.0_30.0_DR0.1_0.1_0.1',
              #r'/hdd6/Models/Inria_GAN/UnetGAN_inria_gan_soft_1_2_PS(572, 572)_BS4_EP30_LR0.0001_1e-05_1e-06_DS30.0_10.0_30.0_DR0.1_0.1_0.1',
              #r'/hdd6/Models/Inria_GAN/0823/UnetGAN_inria_gan_0821_0_2_PS(572, 572)_BS4_EP50_LR0.0001_1e-05_1e-06_DS30.0_10.0_30.0_DR0.1_0.1_0.1',
              #r'/hdd6/Models/Inria_GAN/0823/UnetGAN_inria_gan_0821_1_2_PS(572, 572)_BS4_EP50_LR0.0001_1e-05_1e-06_DS30.0_10.0_30.0_DR0.1_0.1_0.1',
              #r'/hdd6/Models/Inria_GAN/0823/UnetGAN_inria_gan_0821_2_2_PS(572, 572)_BS4_EP50_LR0.0001_1e-05_1e-06_DS30.0_10.0_30.0_DR0.1_0.1_0.1',
              #r'/hdd6/Models/Inria_GAN/0823/UnetGAN_inria_gan_0821_3_2_PS(572, 572)_BS4_EP50_LR0.0001_1e-05_1e-06_DS30.0_10.0_30.0_DR0.1_0.1_0.1',
              #r'/hdd6/Models/Inria_GAN/0823/UnetGAN_inria_gan_0821_4_2_PS(572, 572)_BS4_EP50_LR0.0001_1e-05_1e-06_DS30.0_10.0_30.0_DR0.1_0.1_0.1',
              #r'/hdd6/Models/Inria_GAN/0955pm/UnetGAN_inria_gan_1001am_1_2_PS(572, 572)_BS4_EP30_LR0.0001_0.0001_0.0001_DS15.0_10.0_20.0_DR0.1_0.1_0.1',
              #r'/hdd6/Models/Inria_GAN/0955pm/UnetGAN_inria_gan_1001am_1_2_PS(572, 572)_BS4_EP30_LR0.0001_0.0001_0.0001_DS15.0_10.0_10.0_DR0.1_0.1_0.1',
              #r'/hdd6/Models/Inria_GAN/baseline/UnetGAN_inria_gan_cheat_0_2_PS(572, 572)_BS4_EP50_LR0.0001_1e-05_1e-06_DS30.0_10.0_30.0_DR0.1_0.1_0.1',
              #r'/hdd6/Models/Inria_GAN/baseline/UnetGAN_inria_gan_cheat_1_2_PS(572, 572)_BS4_EP50_LR0.0001_1e-05_1e-06_DS30.0_10.0_30.0_DR0.1_0.1_0.1',
              #r'/hdd6/Models/Inria_GAN/baseline/UnetGAN_inria_gan_cheat_2_2_PS(572, 572)_BS4_EP50_LR0.0001_1e-05_1e-06_DS30.0_10.0_30.0_DR0.1_0.1_0.1',
              #r'/hdd6/Models/Inria_GAN/baseline/UnetGAN_inria_gan_cheat_3_2_PS(572, 572)_BS4_EP50_LR0.0001_1e-05_1e-06_DS30.0_10.0_30.0_DR0.1_0.1_0.1',
              #r'/hdd6/Models/Inria_GAN/baseline/UnetGAN_inria_gan_cheat_4_2_PS(572, 572)_BS4_EP50_LR0.0001_1e-05_1e-06_DS30.0_10.0_30.0_DR0.1_0.1_0.1',
              #r'/hdd6/Models/Inria_GAN/V2/UnetGAN_V2_inria_gan_base_1_2_PS(572, 572)_BS4_EP30_LR0.0001_1e-05_1e-06_DS30.0_10.0_10.0_DR0.1_0.1_0.1',
              r'/hdd6/Models/Inria_GAN/V2/UnetGAN_V2_inria_gan_base_1_2_PS(572, 572)_BS4_EP30_LR0.0001_1e-05_1e-05_DS30.0_10.0_10.0_DR0.1_0.1_0.1',
              ]

for model_dir in model_list:
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
    model = uabMakeNetwork_UNet.UnetModelGAN_V2({'X': X, 'Y': y},
                                             trainable=mode,
                                             input_size=input_size,
                                             batch_size=batch_size,
                                             start_filter_num=32)
    # create graph
    model.create_graph(['X', 'Y'], class_num=2)

    # evaluate on tiles
    model.evaluate(file_list_valid, file_list_valid_truth, parent_dir, parent_dir_truth,
                   input_size, tile_size, batch_size, img_mean, model_dir, gpu,
                   save_result_parent_dir='ugan', ds_name='inria', best_model=False)
