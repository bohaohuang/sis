import os
import cv2
import imageio
import scipy.misc
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import utils
import uabDataReader
import util_functions
import uabUtilreader
from bohaoCustom import uabMakeNetwork_UNet


def get_sum_of_channel(img):
    means = []
    for i in range(3):
        means.append(np.sum(img[:, :, i]))
    return np.array(means)


util_functions.tf_warn_level(3)
adjust_save_dir = r'/media/ei-edl01/data/uab_datasets/sp/data_gamma_adjust'
sp_model_dir = r'/media/ei-edl01/data/uab_datasets/sp/]shared_models/UnetCropCV_(FixedRes)CTFinetune+nlayer9_' \
             r'PS(572, 572)_BS5_EP100_LR1e-05_DS50_DR0.1_SFN32'
mask_save_dir = r'/media/ei-edl01/user/as667/BOHAO/gbdx_mask'
orig_img_dir = r'/media/ei-edl01/user/as667'
task_id = 'Honolulu_chunks'
img_id = '10400100232FC300_1'

gpu = 0
gamma = 2.5
invGamma = 1.0 / gamma
table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype('uint8')
input_size = [1052, 1052]
tile_size = [2541, 2541]

orig_img_name = os.path.join(orig_img_dir, task_id, '{}.tif'.format(img_id))
orig_img = imageio.imread(orig_img_name)
orig_img = scipy.misc.imresize(orig_img, [2541, 2541])
orig_img = cv2.LUT(orig_img, table)
imageio.imsave(os.path.join(adjust_save_dir, '{}.tif'.format(img_id)), orig_img)
img_mean = get_sum_of_channel(orig_img) / (2541 ** 2)
print(img_mean)

tf.reset_default_graph()
# make the model
# define place holder
X = tf.placeholder(tf.float32, shape=[None, input_size[0], input_size[1], 3], name='X')
y = tf.placeholder(tf.int32, shape=[None, input_size[0], input_size[1], 1], name='y')
mode = tf.placeholder(tf.bool, name='mode')
model = uabMakeNetwork_UNet.UnetModelCrop({'X': X, 'Y': y},
                                          trainable=mode,
                                          input_size=input_size,
                                          batch_size=5, start_filter_num=32)
# create graph
model.create_graph('X', class_num=2)
reader = uabDataReader.ImageLabelReader(gtInds=[0],
                                        dataInds=[0],
                                        nChannels=3,
                                        parentDir=adjust_save_dir,
                                        chipFiles=[['{}.tif'.format(img_id)]],
                                        chip_size=input_size,
                                        tile_size=tile_size,
                                        batchSize=5,
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

mask_img = np.copy(orig_img)
mask_img = util_functions.add_mask(mask_img, pred, [255, None, None], mask_1=1)
plt.imshow(mask_img)
plt.show()

plt.imshow(mask_img[1350:1750, 400:1000, :])
plt.show()

imageio.imsave(os.path.join(mask_save_dir, '{}_{}_orig_large.png'.format(task_id, img_id)), orig_img[1350:1750, 400:1000, :])
imageio.imsave(os.path.join(mask_save_dir, '{}_{}_mask_large.png'.format(task_id, img_id)), mask_img[1350:1750, 400:1000, :])
