import os
import cv2
import imageio
import numpy as np
import tensorflow as tf
from glob import glob
from tqdm import tqdm
import uab_collectionFunctions
import util_functions
from bohaoCustom import uabMakeNetwork_UNet


def get_sum_of_channel(img):
    means = []
    for i in range(3):
        means.append(np.sum(img[:, :, i]))
    return np.array(means)


# settings
gpu = 1
batch_size = 5
input_size = [572, 572]
tile_size = [2541, 2541]
util_functions.tf_warn_level(3)
file_dir = r'/media/ei-edl01/data/uab_datasets/sp/DATA_BUILDING_AND_PANEL'
adjust_save_dir = r'/media/ei-edl01/data/uab_datasets/sp/data_gamma_adjust'
model_dir = r'/hdd6/Models/UNET_rand_gird/UnetCrop_inria_aug_grid_0_PS(572, 572)_BS5_EP100_LR0.0001_DS60_DR0.1_SFN32'
imgs = sorted(glob(os.path.join(file_dir, '*.jpg')))
n = len(imgs) * 2541 ** 2

for gamma in [0.2, 0.5, 1, 1.5, 2, 2.5, 3]:
    tf.reset_default_graph()
    img_mean = np.zeros(3)

    # make new imgs
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype('uint8')
    for file in tqdm(imgs):
        img = imageio.imread(file)
        img_adjust = cv2.LUT(img, table)
        img_mean += get_sum_of_channel(img_adjust)
        img_name = os.path.basename(file)
        imageio.imsave(os.path.join(adjust_save_dir, img_name), img_adjust)

    img_mean = img_mean / n
    print(img_mean)

    file_list_valid = [[os.path.basename(x)] for x in sorted(glob(os.path.join(adjust_save_dir, '*.jpg')))]
    file_list_valid_truth = [os.path.basename(x) for x in sorted(glob(os.path.join(file_dir, '*.png')))]

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

    # evaluate on tiles
    model.evaluate(file_list_valid, file_list_valid_truth, adjust_save_dir, file_dir,
                   input_size, tile_size, batch_size, img_mean, model_dir, gpu,
                   save_result_parent_dir='ct_gamma', ds_name='ct{}'.format(util_functions.d2s(gamma)))
