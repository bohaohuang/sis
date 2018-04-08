import os
import cv2
import imageio
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
gpu = 1
batch_size = 5
input_size = [572, 572]
tile_size = [2541, 2541]
util_functions.tf_warn_level(3)
catId = '104001001099F800'
file_dir = r'/media/ei-edl01/user/as667/{}'.format(catId)
adjust_save_dir = r'/media/ei-edl01/data/uab_datasets/sp/data_gamma_adjust'
model_dir = r'/hdd6/Models/UNET_rand_gird/UnetCrop_inria_aug_grid_0_PS(572, 572)_BS5_EP100_LR0.0001_DS60_DR0.1_SFN32'
model_dir2 = r'/media/ei-edl01/data/uab_datasets/sp/]shared_models/UnetCropCV_(FixedRes)CTFinetune+nlayer9_' \
             r'PS(572, 572)_BS5_EP100_LR1e-05_DS50_DR0.1_SFN32'
imgs = sorted(glob(os.path.join(file_dir, '*.tif')))[:-1]
n = len(imgs) * 2541 ** 2

for gamma in [1.5, 1.75, 2, 2.25, 2.5, 2.75, 3]:
    tf.reset_default_graph()
    img_mean = np.zeros(3)

    # make new imgs
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype('uint8')
    for file in tqdm(imgs):
        try:
            img = imageio.imread(file)
        except IndexError:
            print(file)
        img_adjust = cv2.LUT(img, table)
        img_mean += get_sum_of_channel(img_adjust)
        img_name = os.path.basename(file)
        imageio.imsave(os.path.join(adjust_save_dir, img_name), img_adjust)

    img_mean = img_mean / n
    print(img_mean)

    file_list_valid = [[os.path.basename(x)] for x in sorted(glob(os.path.join(adjust_save_dir, '*.tif')))]

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
    for large_tile in file_list_valid:
        reader = uabDataReader.ImageLabelReader(gtInds=[0],
                                                dataInds=[0],
                                                nChannels=3,
                                                parentDir=adjust_save_dir,
                                                chipFiles=[large_tile],
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
            model.load(model_dir, sess)
            result = model.test('X', sess, test_reader)
        image_pred = uabUtilreader.un_patchify_shrink(result,
                                                      [tile_size[0] + model.get_overlap(),
                                                       tile_size[1] + model.get_overlap()],
                                                      tile_size, input_size,
                                                      [input_size[0] - model.get_overlap(),
                                                       input_size[1] - model.get_overlap()],
                                                      overlap=model.get_overlap())
        pred = util_functions.get_pred_labels(image_pred) * 255

        reader = uabDataReader.ImageLabelReader(gtInds=[0],
                                                dataInds=[0],
                                                nChannels=3,
                                                parentDir=adjust_save_dir,
                                                chipFiles=[large_tile],
                                                chip_size=input_size,
                                                tile_size=tile_size,
                                                batchSize=batch_size,
                                                block_mean=img_mean,
                                                overlap=model.get_overlap(),
                                                padding=np.array((model.get_overlap() / 2, model.get_overlap() / 2)),
                                                isTrain=False)
        test_reader = reader.readManager
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            model.load(model_dir2, sess)
            result = model.test('X', sess, test_reader)
        image_pred = uabUtilreader.un_patchify_shrink(result,
                                                      [tile_size[0] + model.get_overlap(),
                                                       tile_size[1] + model.get_overlap()],
                                                      tile_size, input_size,
                                                      [input_size[0] - model.get_overlap(),
                                                       input_size[1] - model.get_overlap()],
                                                      overlap=model.get_overlap())
        pred_sp = util_functions.get_pred_labels(image_pred) * 255

        rgb_img = imageio.imread(os.path.join(adjust_save_dir, large_tile[0]))
        plt.figure(figsize=(14, 5))
        ax1 = plt.subplot(131)
        plt.imshow(rgb_img)
        plt.title('Gamma={}'.format(gamma))
        plt.axis('off')
        ax2 = plt.subplot(132, sharex=ax1, sharey=ax1)
        plt.imshow(pred)
        plt.title('building')
        plt.axis('off')
        ax3 = plt.subplot(133, sharex=ax1, sharey=ax1)
        plt.imshow(pred_sp)
        plt.title('sp')
        plt.axis('off')
        plt.tight_layout()
        img_name = '{}{}.png'.format(large_tile[0].split('.')[0], gamma)
        save_dir = r'/media/ei-edl01/user/bh163/figs/2018.03.15.gbdx/{}'.format(catId)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(os.path.join(save_dir, img_name))
        #plt.show()
