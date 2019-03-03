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
    plt.title('{} IoU={:.2f}'.format(title_str, util_functions.iou_metric(truth, pred)* 100))
    plt.box(True)
    plt.xticks([])
    plt.yticks([])


def make_cmp_plot(rgb, truth, base, loo, mtl, x, y, window_size, city_str):
    rgb_patch = rgb[x:x+window_size, y:y+window_size, :]
    truth_patch = truth[x:x+window_size, y:y+window_size]
    base_patch = make_error_image(truth[x:x+window_size, y:y+window_size], base[x:x+window_size, y:y+window_size])
    loo_patch = make_error_image(truth[x:x+window_size, y:y+window_size], loo[x:x+window_size, y:y+window_size])
    mtl_patch = make_error_image(truth[x:x+window_size, y:y+window_size], mtl[x:x+window_size, y:y+window_size])

    fig = plt.figure(figsize=(12, 3.5))
    plt.subplot(141)
    plt.imshow(rgb_patch)
    plt.title(city_str)
    plt.axis('off')
    plt.subplot(142)
    make_subplot(base_patch, base, truth_patch, 'Base')
    plt.subplot(143)
    make_subplot(loo_patch, loo, truth_patch, 'LOO')
    plt.subplot(144)
    make_subplot(mtl_patch, mtl, truth_patch, 'UGAN')
    plt.tight_layout()

    return fig


cnn_name = 'unet'
top_patch_check = 15
window_size = 500
stride = 200
city_list = ['austin', 'chicago', 'kitsap', 'tyrol-w', 'vienna']
img_dir, task_dir = sis_utils.get_task_img_folder()

truth_dir = r'/media/ei-edl01/data/uab_datasets/inria/data/Original_Tiles'
if cnn_name == 'unet':
    base_dir = r'/hdd/Results/domain_selection/UnetCrop_inria_aug_grid_0_PS(572, 572)_BS5_' \
               r'EP100_LR0.0001_DS60_DR0.1_SFN32/inria/pred'
    loo_dir = r'/hdd/Results/domain_selection/UnetCrop_inria_aug_leave_{}_0_PS(572, 572)_BS5_' \
              r'EP100_LR0.0001_DS60_DR0.1_SFN32/inria/pred'
    mtl_dir = r'/hdd/Results/ugan/UnetGAN_V3Shrink_inria_gan_loo_{}_0_PS(572, 572)_BS20_' \
              r'EP30_LR0.0001_1e-06_1e-06_DS15.0_30.0_30.0_DR0.1_0.1_0.1/inria/pred'
else:
    base_dir = r'/hdd/Results/domain_selection/DeeplabV3_inria_aug_grid_0_PS(321, 321)_BS5_' \
               r'EP100_LR1e-05_DS40_DR0.1_SFN32/inria/pred'
    loo_dir = r'/hdd/Results/domain_selection/DeeplabV3_inria_aug_train_leave_{}_0_PS(321, 321)_BS5_' \
              r'EP100_LR1e-05_DS40_DR0.1_SFN32/inria/pred'
    mtl_dir = None

for city_num in [3, 4]:
    for val_img_cnt in range(1, 6):
        img_save_dir = os.path.join(img_dir, 'uganv3shrink_loo', city_list[city_num])
        if not os.path.exists(img_save_dir):
            os.makedirs(img_save_dir)

        rgb = imageio.imread(os.path.join(truth_dir, '{}{}_RGB.tif'.format(city_list[city_num], val_img_cnt)))

        truth_img_name = os.path.join(truth_dir, '{}{}_GT.tif'.format(city_list[city_num], val_img_cnt))
        base_img_name = os.path.join(base_dir, '{}{}.png'.format(city_list[city_num], val_img_cnt))
        loo_img_name = os.path.join(loo_dir.format(city_num), '{}{}.png'.format(city_list[city_num], val_img_cnt))
        mtl_img_name = os.path.join(mtl_dir.format(city_num), '{}{}.png'.format(city_list[city_num], val_img_cnt))

        truth, base, loo, mtl = imageio.imread(truth_img_name), imageio.imread(base_img_name), \
                                imageio.imread(loo_img_name), imageio.imread(mtl_img_name)
        truth = truth / 255
        base = base / 255

        assert np.all(np.unique(truth) == np.array([0, 1])) and np.all(np.unique(base) == np.array([0, 1])) \
               and np.all(np.unique(loo) == np.array([0, 1])) and np.all(np.unique(mtl) == np.array([0, 1]))

        erp_base, erp_dict_base = error_region_proposals(base, truth, window_size, stride)
        erp_loo, erp_dict_loo = error_region_proposals(loo, truth, window_size, stride)
        erp_mtl, erp_dict_mtl = error_region_proposals(mtl, truth, window_size, stride)

        for patch_cnt in range(top_patch_check):
            x, y = erp_loo[patch_cnt, :][:2]
            fig = make_cmp_plot(rgb, truth, base, loo, mtl, x, y, window_size,
                                '{}{}'.format(city_list[city_num], val_img_cnt))
            plt.savefig(os.path.join(img_save_dir, '{}{}_h{}_w{}.png'.format(city_list[city_num], val_img_cnt, x, y)))
            plt.close(fig)
