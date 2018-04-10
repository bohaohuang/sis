import os
import cv2
import imageio
import scipy.misc
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
import uabDataReader
import uabUtilreader
import util_functions
from bohaoCustom import uabMakeNetwork_UNet


def get_sum_of_channel(img):
    means = []
    for i in range(3):
        means.append(np.sum(img[:, :, i]))
    return np.array(means)


# settings
util_functions.tf_warn_level(3)
task_id = '1040010021B61200'
img_dir = r'/media/ei-edl01/user/as667/{}_rechunked'.format(task_id)
my_dir = r'/media/ei-edl01/user/as667/BOHAO/gbdx_results/{}'.format(task_id)
adjust_save_dir = r'/media/ei-edl01/data/uab_datasets/sp/data_gamma_adjust'
building_model_dir = r'/hdd6/Models/UNET_rand_gird/UnetCrop_inria_aug_grid_0_PS(572, 572)_' \
                     r'BS5_EP100_LR0.0001_DS60_DR0.1_SFN32'
sp_model_dir = r'/media/ei-edl01/data/uab_datasets/sp/]shared_models/UnetCropCV_(FixedRes)CTFinetune+nlayer9_' \
             r'PS(572, 572)_BS5_EP100_LR1e-05_DS50_DR0.1_SFN32'
imgs = sorted(glob(os.path.join(img_dir, '*.tif')))
gamma = 2.5
invGamma = 1.0 / gamma
table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype('uint8')
n = len(imgs) * 2541 ** 2
gpu = 1
batch_size = 5
input_size = [1052, 1052]
tile_size = [2541, 2541]

# make dirs
if not os.path.exists(my_dir):
    os.makedirs(my_dir)

# calculate img_mean
img_mean = np.zeros(3)
img_file = []
for file in tqdm(imgs):
    img_name = os.path.basename(file)
    try:
        img = imageio.imread(file)
        img = scipy.misc.imresize(img, tile_size)
        img_adjust = cv2.LUT(img, table)
        img_mean += get_sum_of_channel(img_adjust)
        img_file.append(file)
        imageio.imsave(os.path.join(adjust_save_dir, img_name), img_adjust)
    except IndexError:
        print(file)
img_mean = img_mean / n
print(img_mean)

# run detector
file_list_valid = [[os.path.basename(x)] for x in img_file]
# make the model
# define place holder
X = tf.placeholder(tf.float32, shape=[None, input_size[0], input_size[1], 3], name='X')
y = tf.placeholder(tf.int32, shape=[None, input_size[0], input_size[1], 1], name='y')
mode = tf.placeholder(tf.bool, name='mode')
model = uabMakeNetwork_UNet.UnetModelCrop({'X':X, 'Y':y},
                                          trainable=mode,
                                          input_size=input_size,
                                          batch_size=5, start_filter_num=32)
# create graph
model.create_graph('X', class_num=2)

# load data
for test_file in file_list_valid:
    tf.reset_default_graph()

    # building detector
    file_name = 'building_' + test_file[0]
    if os.path.exists(os.path.join(my_dir, file_name)):
        continue
    else:
        reader = uabDataReader.ImageLabelReader(gtInds=[0],
                                                dataInds=[0],
                                                nChannels=3,
                                                parentDir=adjust_save_dir,
                                                chipFiles=[test_file],
                                                chip_size=input_size,
                                                tile_size=tile_size,
                                                batchSize=batch_size,
                                                block_mean=img_mean,
                                                overlap=model.get_overlap(),
                                                padding=np.array((model.get_overlap() / 2, model.get_overlap() / 2)),
                                                isTrain=False)
        test_reader = reader.readManager
        # run algo
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            model.load(building_model_dir, sess)
            result = model.test('X', sess, test_reader)
        image_pred = uabUtilreader.un_patchify_shrink(result,
                                                      [tile_size[0] + model.get_overlap(),
                                                       tile_size[1] + model.get_overlap()],
                                                      tile_size, input_size,
                                                      [input_size[0] - model.get_overlap(),
                                                       input_size[1] - model.get_overlap()],
                                                      overlap=model.get_overlap())
        pred = util_functions.get_pred_labels(image_pred)
        imageio.imsave(os.path.join(my_dir, file_name), pred)

    # sp detector
    file_name = 'sp_' + test_file[0]
    if os.path.exists(os.path.join(my_dir, file_name)):
        continue
    else:
        reader = uabDataReader.ImageLabelReader(gtInds=[0],
                                                dataInds=[0],
                                                nChannels=3,
                                                parentDir=adjust_save_dir,
                                                chipFiles=[test_file],
                                                chip_size=input_size,
                                                tile_size=tile_size,
                                                batchSize=batch_size,
                                                block_mean=img_mean,
                                                overlap=model.get_overlap(),
                                                padding=np.array((model.get_overlap() / 2, model.get_overlap() / 2)),
                                                isTrain=False)
        test_reader = reader.readManager
        # run algo
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            model.load(sp_model_dir, sess)
            result = model.test('X', sess, test_reader)
        image_pred = uabUtilreader.un_patchify_shrink(result,
                                                      [tile_size[0] + model.get_overlap(),
                                                       tile_size[1] + model.get_overlap()],
                                                      tile_size, input_size,
                                                      [input_size[0] - model.get_overlap(),
                                                       input_size[1] - model.get_overlap()],
                                                      overlap=model.get_overlap())
        pred = util_functions.get_pred_labels(image_pred)
        imageio.imsave(os.path.join(my_dir, file_name), pred)
