import tensorflow as tf
import uab_collectionFunctions
import util_functions
import uabCrossValMaker
from bohaoCustom import uabMakeNetwork_UNet

# settings
gpu = 1
batch_size = 5
input_size = [572, 572]
tile_size = [1500, 1500]
model_type = 'gan'
util_functions.tf_warn_level(3)

city_name = 'Mass_road'
# model_dir = r'/hdd6/Models/Inria_GAN/Road/UnetCrop_road_0_PS(572, 572)_BS5_EP80_LR0.0001_DS60_DR0.1_SFN32'
model_dir = r'/hdd6/Models/Inria_GAN/Road/UnetGAN_V3ShrinkRGB_road_xregion_gan_0_PS(572, 572)_BS20_EP60_LR0.0001_1e-06_1e-06_DS60.0_60.0_60.0_DR0.1_0.1_0.1'
blCol = uab_collectionFunctions.uabCollection(city_name)
blCol.readMetadata()
file_list, parent_dir = blCol.getAllTileByDirAndExt([0, 1, 2])
file_list_truth, parent_dir_truth = blCol.getAllTileByDirAndExt(4)
img_mean = blCol.getChannelMeans([0, 1, 2])

# use uabCrossValMaker to get fileLists for training and validation
idx, file_list = uabCrossValMaker.uabUtilGetFolds(None, file_list, 'city')
idx_truth, file_list_truth = uabCrossValMaker.uabUtilGetFolds(None, file_list_truth, 'city')
file_list_test = uabCrossValMaker.make_file_list_by_key(idx, file_list, [0])
file_list_test_truth = uabCrossValMaker.make_file_list_by_key(idx_truth, file_list_truth, [0])

# make the model
# define place holder
X = tf.placeholder(tf.float32, shape=[None, input_size[0], input_size[1], 3], name='X')
y = tf.placeholder(tf.int32, shape=[None, input_size[0], input_size[1], 1], name='y')
mode = tf.placeholder(tf.bool, name='mode')
if model_type == 'unet':
    model = uabMakeNetwork_UNet.UnetModelCrop({'X': X, 'Y': y},
                                              trainable=mode,
                                              input_size=input_size,
                                              batch_size=batch_size,
                                              start_filter_num=32)
    # create graph
    model.create_graph('X', class_num=2)
else:
    model = uabMakeNetwork_UNet.UnetModelGAN_V3ShrinkRGB({'X': X, 'Y': y},
                                                         trainable=mode,
                                                         input_size=input_size,
                                                         batch_size=batch_size,
                                                         start_filter_num=32)
    # create graph
    model.create_graph(['X', 'Y'], class_num=2)

# evaluate on tiles
model.evaluate(file_list_test, file_list_test_truth, parent_dir, parent_dir_truth,
               input_size, tile_size, batch_size, img_mean, model_dir, gpu,
               save_result_parent_dir='Road', ds_name=city_name, best_model=True)
