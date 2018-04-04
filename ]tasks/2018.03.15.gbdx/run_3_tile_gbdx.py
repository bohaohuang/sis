import os
import cv2
import imageio
import scipy.misc
import scipy.ndimage
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from glob import glob
import utils
import uabRepoPaths
import util_functions
from util_functions import add_mask
from bohaoCustom import uabMakeNetwork_UNet


def get_sum_of_channel(img):
    means = []
    for i in range(3):
        means.append(np.sum(img[:, :, i]))
    return np.array(means)


tile_dir = r'/media/ei-edl01/user/as667/BOHAO'
orig_dir = r'/media/ei-edl01/data/uab_datasets/sp/DATA_BUILDING_AND_PANEL'
adjust_save_dir = r'/media/ei-edl01/user/as667/BOHAO/adjust_gamma'
model_dir = r'/media/ei-edl01/data/uab_datasets/sp/]shared_models/UnetCropCV_(FixedRes)CTFinetune+nlayer9_' \
            r'PS(572, 572)_BS5_EP100_LR1e-05_DS50_DR0.1_SFN32'
reshape_files = glob(os.path.join(tile_dir, 'rs_*.tif'))
gpu = 1
batch_size = 5
input_size = [572, 572]
tile_size = [2541, 2541]
util_functions.tf_warn_level(3)
img_dir, task_dir = utils.get_task_img_folder()

if len(reshape_files) == 0:
    files = glob(os.path.join(tile_dir, '*.tif'))
    for file in files:
        img = imageio.imread(file)
        new_file_name = os.path.join(tile_dir, 'rs_'+os.path.basename(file))
        img = scipy.misc.imresize(img, (2541, 2541))
        imageio.imsave(new_file_name, img)
    reshape_files = glob(os.path.join(tile_dir, 'rs_*.tif'))

tile_ids = [os.path.basename(a)[3:].split('.')[0] for a in reshape_files]
#gamma_list = np.arange(0.1, 5.1, 0.1) #[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.5, 1.8, 2, 2.5, 3]
gamma_list = [1]
gt_target_pixels = np.zeros((len(tile_ids), len(gamma_list)))
pred_target_pixels = np.zeros((len(tile_ids), len(gamma_list)))
gt_target_num = np.zeros((len(tile_ids), len(gamma_list)))
pred_target_num = np.zeros((len(tile_ids), len(gamma_list)))
for cnt_1, gamma in enumerate(gamma_list):
    path = os.path.join(uabRepoPaths.evalPath, 'sp_gamma', 'UnetCropCV_(FixedRes)CTFinetune+nlayer9_'
                                                                         'PS(572, 572)_BS5_EP100_LR1e-05_DS50_DR0.1_'
                                                                         'SFN32',
                                       'ct{}'.format(util_functions.d2s(gamma)))

    # make new imgs
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype('uint8')
    for rs_file, tile_id in zip(reshape_files, tile_ids):
        img_rs = imageio.imread(rs_file)
        img_adjust = cv2.LUT(img_rs, table)
        img_name = os.path.basename(rs_file)[3:].split('.')[0].replace('_', '-') + '_RGB.jpg'
        imageio.imsave(os.path.join(adjust_save_dir, img_name), img_adjust)

    if not os.path.exists(path):
        print(path)
        tf.reset_default_graph()
        img_mean = np.zeros(3)
        n = len(reshape_files) * 2541 ** 2
        # make new imgs
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype('uint8')
        for rs_file, tile_id in zip(reshape_files, tile_ids):
            img_rs = imageio.imread(rs_file)
            img_or = imageio.imread(os.path.join(orig_dir, tile_id.replace('_', '-')+'_RGB.jpg'))
            gt = imageio.imread(os.path.join(orig_dir, tile_id.replace('_', '-')+'_GT.png'))

            img_adjust = cv2.LUT(img_rs, table)
            img_mean += get_sum_of_channel(img_adjust)
            img_name = os.path.basename(rs_file)[3:].split('.')[0].replace('_', '-')+'_RGB.jpg'
            imageio.imsave(os.path.join(adjust_save_dir, img_name), img_adjust)

        img_mean = img_mean / n
        print(gamma, img_mean)
        file_list_valid = [[os.path.basename(x)] for x in sorted(glob(os.path.join(adjust_save_dir, '*.jpg')))]
        file_list_valid_truth = [os.path.basename(x) for x in sorted(glob(os.path.join(tile_dir, '*.png')))]

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

        # evaluate on tiles
        model.evaluate(file_list_valid, file_list_valid_truth, adjust_save_dir, tile_dir,
                       input_size, tile_size, batch_size, img_mean, model_dir, gpu,
                       save_result_parent_dir='sp_gamma', ds_name='ct{}'.format(util_functions.d2s(gamma)))

    # evaluate performance
    for cnt_2, tile_id in enumerate(tile_ids):
        pred_file = os.path.join(path, 'pred', tile_id.replace('_', '-')+'.png')
        gt_file = os.path.join(orig_dir, tile_id.replace('_', '-') + '_GT.png')
        rgb_file = os.path.join(orig_dir, tile_id.replace('_', '-') + '_RGB.jpg')
        adjust_file = os.path.join(adjust_save_dir, tile_id.replace('_', '-') + '_RGB.jpg')
        p = imageio.imread(pred_file)
        g = imageio.imread(gt_file)
        rgb_orig = imageio.imread(rgb_file)
        rgb_adju = imageio.imread(adjust_file)

        _, cc_p = scipy.ndimage.label(p)
        _, cc_g = scipy.ndimage.label(g)

        pred_target_pixels[cnt_2, cnt_1] = np.sum(p)
        gt_target_pixels[cnt_2, cnt_1] = np.sum(g)
        pred_target_num[cnt_2, cnt_1] = cc_p
        gt_target_num[cnt_2, cnt_1] = cc_g

        masked_orig = add_mask(rgb_orig, g, [255, None, None], mask_1=1)
        masked_adju = add_mask(rgb_adju, p, [255, None, None], mask_1=1)

        plt.figure(figsize=(12, 6))
        ax1 = plt.subplot(121)
        plt.imshow(masked_orig)
        plt.axis('off')
        plt.title('CT')
        ax2 = plt.subplot(122, sharex=ax1, sharey=ax1)
        plt.imshow(masked_adju)
        plt.axis('off')
        plt.title('GBDX')
        plt.suptitle('{}'.format(tile_id))
        plt.tight_layout()
        plt.show()


'''plt.figure(figsize=(14, 5))
for i in range(len(tile_ids)):
    plt.subplot(231+i)
    plt.plot(gamma_list, pred_target_pixels[i, :], '-o', label='pred')
    plt.plot(gamma_list, gt_target_pixels[i, :], '--.', label='gt')
    plt.title(tile_ids[i])
    plt.grid(True)

    plt.subplot(234 + i)
    plt.plot(gamma_list, pred_target_num[i, :], '-o', label='pred')
    plt.plot(gamma_list, gt_target_num[i, :], '--.', label='gt')
    plt.title(tile_ids[i])
    plt.grid(True)
plt.tight_layout()
#plt.savefig(os.path.join(img_dir, 'iou_adjust_cmp.png'))
#plt.show()

plt.figure(figsize=(6, 4))
obj_err = np.sum(np.abs(pred_target_num - gt_target_num), axis=0)
plt.plot(gamma_list, obj_err, '-o')
plt.xlabel('Gamma')
plt.ylabel('abs(sum(err))')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'iou_adjust_err_curve.png'))
plt.show()'''
