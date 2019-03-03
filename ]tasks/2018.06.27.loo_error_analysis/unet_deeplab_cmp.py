import os
import imageio
import numpy as np
import matplotlib.pyplot as plt
import sis_utils
import util_functions


def error_region_proposals(pred, truth, window_size=300, stride=150):
    h, w = pred.shape

    diff = pred - truth
    erp = []
    erp_dict = {}

    for i in range(0, h-window_size, stride):
        for j in range(0, w-window_size, stride):
            err_num = np.sum(np.abs(diff[i:i+window_size, j:j+window_size]))
            erp.append([i, j, err_num])
            erp_dict['h{}w{}'.format(i, j)] = err_num
    erp = np.array(erp, dtype=int)
    erp = erp[erp[:, 2].argsort()][::-1]
    return erp, erp_dict


def make_error_image(truth, pred):
    h, w = truth.shape
    patch_img = 255 * np.ones((h, w, 3), dtype=np.uint8)
    mask = truth - pred
    patch_img = util_functions.add_mask(patch_img, truth, [0, 255, 0], 1)
    patch_img = util_functions.add_mask(patch_img, mask, [0, 0, 255], 1)
    patch_img = util_functions.add_mask(patch_img, mask, [255, 0, 0], -1)
    return patch_img


def make_subplot(patch, pred, truth, title_str):
    plt.imshow(patch)
    pred = pred[x:x + window_size, y:y + window_size]
    fn = np.sum(truth - pred == 1) / (window_size ** 2) * 100
    fp = np.sum(pred - truth == 1) / (window_size ** 2) * 100
    plt.title('{} IoU={:.2f}\nFN={:.2f},FP={:.2f}'.format(title_str, util_functions.iou_metric(truth, pred)* 100,
                                                           fn, fp))
    plt.box(True)
    plt.xticks([])
    plt.yticks([])


def make_cmp_plot(rgb, truth, cnn_base, deeplab_base, x, y, window_size, city_str):
    rgb_patch = rgb[x:x+window_size, y:y+window_size, :]
    truth_patch = truth[x:x+window_size, y:y+window_size]
    cnn_base_patch = make_error_image(truth[x:x+window_size, y:y+window_size], cnn_base[x:x+window_size, y:y+window_size])
    deeplab_base_patch = make_error_image(truth[x:x+window_size, y:y+window_size], deeplab_base[x:x+window_size, y:y+window_size])

    fig = plt.figure(figsize=(9, 3.5))
    plt.subplot(131)
    plt.imshow(rgb_patch)
    plt.title(city_str.title())
    plt.axis('off')
    plt.subplot(132)
    make_subplot(cnn_base_patch, cnn_base, truth_patch, 'U-Net')
    plt.subplot(133)
    make_subplot(deeplab_base_patch, deeplab_base, truth_patch, 'DeepLab')
    plt.tight_layout()

    return fig


cnn_name = 'unet'
top_patch_check = 15
window_size = 300
stride = 200
city_list = ['austin', 'chicago', 'kitsap', 'tyrol-w', 'vienna']
img_dir, task_dir = sis_utils.get_task_img_folder()

truth_dir = r'/media/ei-edl01/data/uab_datasets/inria/data/Original_Tiles'
cnn_base_dir = r'/hdd/Results/domain_selection/UnetCrop_inria_aug_grid_0_PS(572, 572)_BS5_' \
               r'EP100_LR0.0001_DS60_DR0.1_SFN32/inria/pred'
deeplab_base_dir = r'/hdd/Results/domain_selection/DeeplabV3_inria_aug_grid_0_PS(321, 321)_BS5_' \
                   r'EP100_LR1e-05_DS40_DR0.1_SFN32/inria/pred'

for city_num in range(5):
    for val_img_cnt in range(1, 6):
        img_save_dir = os.path.join(img_dir, 'unet_deeplab_cmp', city_list[city_num])
        if not os.path.exists(img_save_dir):
            os.makedirs(img_save_dir)

        rgb = imageio.imread(os.path.join(truth_dir, '{}{}_RGB.tif'.format(city_list[city_num], val_img_cnt)))

        truth_img_name = os.path.join(truth_dir, '{}{}_GT.tif'.format(city_list[city_num], val_img_cnt))
        cnn_base_img_name = os.path.join(cnn_base_dir, '{}{}.png'.format(city_list[city_num], val_img_cnt))
        deeplab_base_img_name = os.path.join(deeplab_base_dir, '{}{}.png'.format(city_list[city_num], val_img_cnt))

        truth, cnn_base, deeplab_base = imageio.imread(truth_img_name), imageio.imread(cnn_base_img_name), \
                                        imageio.imread(deeplab_base_img_name)

        truth = truth / 255
        cnn_base = cnn_base / 255
        deeplab_base = deeplab_base / 255

        assert np.all(np.unique(truth) == np.array([0, 1])) and np.all(np.unique(cnn_base) == np.array([0, 1])) \
               and np.all(np.unique(deeplab_base) == np.array([0, 1]))

        erp, erp_dict = error_region_proposals(cnn_base, deeplab_base, window_size, stride)

        for patch_cnt in range(top_patch_check):
            x, y = erp[patch_cnt, :][:2]
            fig = make_cmp_plot(rgb, truth, cnn_base, deeplab_base, x, y, window_size,
                                '{}{}'.format(city_list[city_num], val_img_cnt))
            plt.savefig(os.path.join(img_save_dir, '{}{}_h{}_w{}.png'.format(city_list[city_num], val_img_cnt, x, y)))
            plt.close(fig)
