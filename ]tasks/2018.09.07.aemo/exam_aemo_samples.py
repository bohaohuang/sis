import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from glob import glob
import ersa_utils
import util_functions
import uabDataReader
import uabUtilreader
from nn import unet, nn_utils
from preprocess import gammaAdjust
from collection import collectionMaker as cm
from bohaoCustom import uabMakeNetwork_UNet


def adjust_gamma(img, gamma):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype('uint8')
    img_adjust = cv2.LUT(img, table)
    return img_adjust


# define parameters
class_num = 2
patch_size = (572, 572)
tile_size = (5000, 5000)
bs = 5
suffix = 'test'
sfn = 32
gpu = 0
nn_utils.set_gpu(gpu)

# get test data
gammas = [2.5, 1, 2.5]
sample_id = 3
data_dir = r'/media/ei-edl01/data/aemo/samples/0584007740{}0_01'.format(sample_id)
files = sorted(glob(os.path.join(data_dir, 'TILES', '*.tif')))

# adjust gamma
gamma_save_dir = os.path.join(data_dir, 'gamma_adjust')
ersa_utils.make_dir_if_not_exist(gamma_save_dir)
ga = gammaAdjust.GammaAdjust(gamma=gammas[sample_id - 1], path=gamma_save_dir)
ga.run(force_run=False, file_list=files)
files = sorted(glob(os.path.join(gamma_save_dir, '*.tif')))

# get image mean
img_mean = cm.get_channel_mean('', [[f] for f in files])

model_dir = r'/hdd6/Models/UNET_city/UnetCrop_spca_aug_xcity_0_PS(572, 572)_BS5_EP100_LR0.0001_DS60_DR0.1_SFN32'

# test model
'''nn_utils.tf_warn_level(3)
unet = unet.UNet(class_num, patch_size, suffix=suffix, batch_size=bs)
model_dir = r'/hdd6/Models/UNET_city/UnetCrop_spca_aug_xcity_0_PS(572, 572)_BS5_EP100_LR0.0001_DS60_DR0.1_SFN32'
# model_dir = r'/hdd6/Models/Inria_decay/UnetCrop_inria_decay_0_PS(572, 572)_BS5_EP100_LR0.0001_DS60.0_DR0.1_SFN32'
unet.evaluate([[f] for f in files], patch_size, tile_size, bs, img_mean, model_dir, gpu, save_result_parent_dir='aemo',
              sfn=sfn, force_run=False, score_results=False, split_char='.', best_model=False)'''


my_dir = os.path.join(data_dir, 'bh_pred')

# make dirs
if not os.path.exists(my_dir):
    os.makedirs(my_dir)

# run detector
file_list_valid = [[os.path.basename(x)] for x in files]

# load data
for test_file in file_list_valid:
    tf.reset_default_graph()
    # make the model
    # define place holder
    X = tf.placeholder(tf.float32, shape=[None, patch_size[0], patch_size[1], 3], name='X')
    y = tf.placeholder(tf.int32, shape=[None, patch_size[0], patch_size[1], 1], name='y')
    mode = tf.placeholder(tf.bool, name='mode')
    model = uabMakeNetwork_UNet.UnetModelCrop({'X': X, 'Y': y},
                                              trainable=mode,
                                              input_size=patch_size,
                                              batch_size=5, start_filter_num=32)
    # create graph
    model.create_graph('X', class_num=2)

    # sp detector
    file_name = 'sp_' + test_file[0]
    if os.path.exists(os.path.join(my_dir, file_name)):
        continue
    else:
        reader = uabDataReader.ImageLabelReader(gtInds=[0],
                                                dataInds=[0],
                                                nChannels=3,
                                                parentDir=gamma_save_dir,
                                                chipFiles=[test_file],
                                                chip_size=patch_size,
                                                tile_size=tile_size,
                                                batchSize=bs,
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
                                                      tile_size, patch_size,
                                                      [patch_size[0] - model.get_overlap(),
                                                       patch_size[1] - model.get_overlap()],
                                                      overlap=model.get_overlap())
        pred = util_functions.get_pred_labels(image_pred)
        file_name = file_name.split('.')[0] + '.tif'
        ersa_utils.save_file(os.path.join(my_dir, file_name), pred)

'''pred = ersa_utils.load_file(os.path.join(data_dir, 'bh_pred', 'sp_aus_0x0_gamma1.tif'))
plt.imshow(pred)
plt.show()'''

