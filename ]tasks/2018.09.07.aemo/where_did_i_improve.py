import os
import imageio
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import sis_utils
import ersa_utils
import util_functions
from visualize import visualize_utils


def error_region_proposals(pred, truth, window_size=300, stride=150):
    h, w = pred.shape

    diff = pred - truth
    erp = []
    erp_dict = {}

    for i in range(0, h - window_size, stride):
        for j in range(0, w - window_size, stride):
            err_num = np.sum(np.abs(diff[i:i + window_size, j:j + window_size]))
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
    iou = util_functions.iou_metric(truth, pred) * 100
    plt.title('{} IoU={:.2f}'.format(title_str, iou))
    plt.box(True)
    plt.xticks([])
    plt.yticks([])
    return iou


def make_cmp_plot(rgb, truth, ft, cmp, x, y, window_size, city_str):
    rgb_patch = rgb[x:x + window_size, y:y + window_size, :]
    truth_patch = truth[x:x + window_size, y:y + window_size]
    ft_patch = make_error_image(truth[x:x + window_size, y:y + window_size], ft[x:x + window_size, y:y + window_size])
    cmp_patch = make_error_image(truth[x:x + window_size, y:y + window_size], cmp[x:x + window_size, y:y + window_size])

    fig = plt.figure(figsize=(10, 3.5))
    plt.subplot(131)
    plt.imshow(rgb_patch)
    plt.title(city_str)
    plt.axis('off')
    plt.subplot(132)
    iou_cmp = make_subplot(cmp_patch, cmp, truth_patch, 'Old')
    plt.subplot(133)
    iou = make_subplot(ft_patch, ft, truth_patch, 'New')
    plt.tight_layout()

    return fig, iou, iou_cmp


if __name__ == '__main__':
    window_size = 200
    stride = 200
    img_dir, task_dir = sis_utils.get_task_img_folder()
    good_th = 60
    bad_th = 20

    truth_dir = r'/media/ei-edl01/data/uab_datasets/aemo/data/Original_Tiles'
    ft_dir = r'/hdd/Results/aemo/uab/UnetCrop_aemo_ft_1_PS(572, 572)_BS5_EP80_LR0.001_DS30_DR0.1_SFN32/aemo/pred'
    cmp_dir = r'/hdd/Results/aemo/uab/UnetCrop_aemo_hd_0_wf3_PS(572, 572)_BS5_EP20_LR1e-05_DS10_DR0.1_SFN32/aemo/pred'

    img_save_dir = os.path.join(img_dir, 'uab_pred_improves_reweight', ft_dir.split('/')[-3])
    ersa_utils.make_dir_if_not_exist(img_save_dir)

    eval_files = ['_'.join(os.path.basename(f)[:-4].split('_')[:2]) for f in glob(os.path.join(ft_dir, '*.png'))]

    for f in eval_files:
        rgb = imageio.imread(os.path.join(truth_dir, '{}_RGB.tif'.format(f)))
        truth_img_name = os.path.join(truth_dir, '{}_GT.tif'.format(f))
        ft_img_name = os.path.join(ft_dir, '{}.png'.format(f))
        cmp_img_name = os.path.join(cmp_dir, '{}.png'.format(f))

        truth, ft, cmp = imageio.imread(truth_img_name), imageio.imread(ft_img_name), imageio.imread(cmp_img_name)

        assert np.all(np.unique(truth) == np.array([0, 1])) and np.all(np.unique(ft) == np.array([0, 1])) and \
               np.all(np.unique(cmp) == np.array([0, 1]))

        erp_ft, erp_dict_ft = error_region_proposals(ft, truth, window_size, stride)

        for patch_cnt in range(len(erp_ft)):
            x, y = erp_ft[patch_cnt, :][:2]
            fig, iou, iou_cmp = make_cmp_plot(rgb, truth, ft, cmp, x, y, window_size, '{}_X{}_Y{}'.format(f, x, y))
            if iou - iou_cmp >= 10:
                plt.savefig(os.path.join(img_save_dir, '{}_h{}_w{}.png'.format(f, y, x)))
            plt.close(fig)
