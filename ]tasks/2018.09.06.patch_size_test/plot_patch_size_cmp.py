import os
import imageio
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import utils
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
    plt.title('{} IoU={:.2f}'.format(title_str, util_functions.iou_metric(truth, pred) * 100))
    plt.box(True)
    plt.xticks([])
    plt.yticks([])


def make_cmp_plot(rgb, truth, base, ugan, x, y, window_size, city_str):
    rgb_patch = rgb[x:x+window_size, y:y+window_size, :]
    truth_patch = truth[x:x+window_size, y:y+window_size]
    base_patch = make_error_image(truth[x:x+window_size, y:y+window_size], base[x:x+window_size, y:y+window_size])
    ugan_patch = make_error_image(truth[x:x+window_size, y:y+window_size], ugan[x:x+window_size, y:y+window_size])

    fig = plt.figure(figsize=(10, 3.5))
    plt.subplot(131)
    plt.imshow(rgb_patch)
    plt.title(city_str)
    plt.axis('off')
    plt.subplot(132)
    make_subplot(base_patch, base, truth_patch, 'Small')
    plt.subplot(133)
    make_subplot(ugan_patch, ugan, truth_patch, 'Large')
    plt.tight_layout()

    return fig


cnn_name = 'deeplab'
top_patch_check = 15
window_size = 500
stride = 200
city_list = ['austin', 'chicago', 'kitsap', 'tyrol-w', 'vienna']
img_dir, task_dir = utils.get_task_img_folder()

truth_dir = r'/media/ei-edl01/data/uab_datasets/Inria/data/Original_Tiles'
if cnn_name == 'unet':
    base_dir = r'/hdd/Results/size/DeeplabV3_inria_decay_0_PS(321, 321)_BS5_EP100_LR1e-05_DS40.0_DR0.1_SFN32/inria_252/pred'
    ugan_dir = r'/hdd/Results/size/DeeplabV3_inria_decay_0_PS(321, 321)_BS5_EP100_LR1e-05_DS40.0_DR0.1_SFN32/inria_892/pred'
else:
    base_dir = r'/hdd/Results/size/DeeplabV3_inria_decay_0_PS(321, 321)_BS5_EP100_LR1e-05_DS40.0_DR0.1_SFN32/inria_252/pred'
    ugan_dir = r'/hdd/Results/size/DeeplabV3_inria_decay_0_PS(321, 321)_BS5_EP100_LR1e-05_DS40.0_DR0.1_SFN32/inria_892/pred'

for city_num in range(5):
    for val_img_cnt in tqdm(range(1, 6)):
        img_save_dir = os.path.join(img_dir, 'patch_size_cmp')
        if not os.path.exists(img_save_dir):
            os.makedirs(img_save_dir)

        rgb = imageio.imread(os.path.join(truth_dir, '{}{}_RGB.tif'.format(city_list[city_num], val_img_cnt)))

        truth_img_name = os.path.join(truth_dir, '{}{}_GT.tif'.format(city_list[city_num], val_img_cnt))
        base_img_name = os.path.join(base_dir, '{}{}.png'.format(city_list[city_num], val_img_cnt))
        ugan_img_name = os.path.join(ugan_dir, '{}{}.png'.format(city_list[city_num], val_img_cnt))

        truth, base, ugan = imageio.imread(truth_img_name), imageio.imread(base_img_name), \
                            imageio.imread(ugan_img_name)
        truth = truth / 255

        assert np.all(np.unique(truth) == np.array([0, 1])) and np.all(np.unique(base) == np.array([0, 1])) \
               and np.all(np.unique(ugan) == np.array([0, 1]))

        erp_base, erp_dict_base = error_region_proposals(base, truth, window_size, stride)
        erp_loo, erp_dict_loo = error_region_proposals(ugan, truth, window_size, stride)

        for patch_cnt in range(top_patch_check):
            x, y = erp_loo[patch_cnt, :][:2]
            fig = make_cmp_plot(rgb, truth, base, ugan, x, y, window_size,
                                '{}{}'.format(city_list[city_num], val_img_cnt))
            plt.savefig(os.path.join(img_save_dir, '{}{}_h{}_w{}.png'.format(city_list[city_num], val_img_cnt, x, y)))
            plt.close(fig)
