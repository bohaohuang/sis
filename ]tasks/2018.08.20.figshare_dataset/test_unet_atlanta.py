import tensorflow as tf
import uabCrossValMaker
import uab_collectionFunctions
import util_functions
from bohaoCustom import uabMakeNetwork_UNet

# settings
gpu = 1
batch_size = 5
input_size = [572, 572]
tile_size = [2500, 2500]
util_functions.tf_warn_level(3)
city_list = ['austin', 'chicago', 'kitsap', 'tyrol-w', 'vienna']

city_name = 'dc'
model_dir = r'/hdd6/Models/UNET_rand_gird/UnetCrop_inria_aug_grid_0_PS(572, 572)_BS5_EP100_LR0.0001_DS60_DR0.1_SFN32'
blCol = uab_collectionFunctions.uabCollection(city_name)
blCol.readMetadata()
file_list, parent_dir = blCol.getAllTileByDirAndExt([0, 1, 2])
file_list_truth, parent_dir_truth = blCol.getAllTileByDirAndExt(3)
img_mean = blCol.getChannelMeans([0, 1, 2])
print(img_mean)

# make the model
# define place holder
X = tf.placeholder(tf.float32, shape=[None, input_size[0], input_size[1], 3], name='X')
y = tf.placeholder(tf.int32, shape=[None, input_size[0], input_size[1], 1], name='y')
mode = tf.placeholder(tf.bool, name='mode')
model = uabMakeNetwork_UNet.UnetModelCrop({'X':X, 'Y':y},
                                          trainable=mode,
                                          input_size=input_size,
                                          batch_size=batch_size,
                                          start_filter_num=32)
# create graph
model.create_graph('X', class_num=2)

# evaluate on tiles
model.evaluate(file_list, file_list_truth, parent_dir, parent_dir_truth,
               input_size, tile_size, batch_size, img_mean, model_dir, gpu,
               save_result_parent_dir='kyle', ds_name=city_name, best_model=False)
