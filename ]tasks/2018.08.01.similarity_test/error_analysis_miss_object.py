import os
import imageio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import measure
from scipy import ndimage
from tqdm import tqdm
import utils
import util_functions


def make_error_image(truth, pred):
    h, w = truth.shape
    patch_img = 255 * np.ones((h, w, 3), dtype=np.uint8)
    mask = truth - pred
    patch_img = util_functions.add_mask(patch_img, truth, [0, 255, 0], 1)
    patch_img = util_functions.add_mask(patch_img, mask, [0, 0, 255], 1)
    patch_img = util_functions.add_mask(patch_img, mask, [255, 0, 0], -1)
    return patch_img


def make_subplot(patch, title_str):
    plt.imshow(patch)
    plt.title('{}'.format(title_str))
    plt.box(True)
    plt.xticks([])
    plt.yticks([])


def make_cmp_plot(rgb, truth, pred, pred_cmp, city_str, rect, building_size):
    pred_patch = make_error_image(truth, pred)
    pred_cmp_patch = make_error_image(truth, pred_cmp)

    fig = plt.figure(figsize=(9, 3.5))
    ax1 = plt.subplot(131)
    plt.imshow(rgb)
    ax1.add_patch(rect[0])
    plt.title('{} Size={}'.format(city_str.title(), int(building_size)))
    plt.axis('off')
    ax2 = plt.subplot(132)
    make_subplot(pred_patch, 'LOO')
    ax2.add_patch(rect[1])
    ax3 = plt.subplot(133)
    make_subplot(pred_cmp_patch, 'MMD')
    ax3.add_patch(rect[2])
    plt.tight_layout()

    return fig


def increase_size(x1, x2, window_size):
    if x2 - x1 < window_size:
        x1 = max(0, x1 - window_size//2)
        x2 = x1 + window_size
    return x1, x2


def make_smallest_region(loc, window_size=200, num=3):
    x1, x2, y1, y2 = loc[0].start, loc[0].stop, loc[1].start, loc[1].stop
    x1_new, x2_new = increase_size(x1, x2, window_size)
    y1_new, y2_new = increase_size(y1, y2, window_size)

    rect = []
    for i in range(num):
        rect.append(patches.Rectangle((y1-y1_new, x1-x1_new), y2-y1, x2-x1, linewidth=2, edgecolor='r',
                                      facecolor='none'))
    return [slice(x1_new, x2_new), slice(y1_new, y2_new)], rect


def visualize_fn_object(rgb, gt, pred, pred_cmp):
    h, w, _ = rgb.shape
    lbl = measure.label(gt)
    building_idx = np.unique(lbl)

    for cnt, idx in enumerate(tqdm(building_idx)):
        on_target = np.sum(pred[np.where(lbl == idx)])
        on_target_cmp = np.sum(pred_cmp[np.where(lbl == idx)])
        building_size = np.sum(gt[np.where(lbl == idx)])
        flag = 0
        if on_target == 0 and on_target_cmp == 0:
            flag = 1
        elif on_target == 0 and on_target_cmp != 0:
            flag = 2
        elif on_target != 0 and on_target_cmp == 0:
            flag = 3
        if flag > 0:
            temp_mask = np.zeros((h, w), dtype=np.uint8)
            temp_mask[np.where(lbl == idx)] = 1
            loc = ndimage.find_objects(temp_mask)[0]
            loc, rect = make_smallest_region(loc)

            fig = make_cmp_plot(rgb[loc], gt[loc], pred[loc], pred_cmp[loc],
                          '{}{}'.format(city_list[city_num], val_img_cnt), rect, building_size)
            yield fig, flag, building_size


city_list = ['austin', 'chicago', 'kitsap', 'tyrol-w', 'vienna']
img_dir, task_dir = utils.get_task_img_folder()
truth_dir = r'/media/ei-edl01/data/uab_datasets/inria/data/Original_Tiles'
cnn_base_dir = r'/hdd/Results/domain_selection/UnetCrop_inria_aug_leave_{}_0_PS(572, 572)_BS5_' \
              r'EP100_LR0.0001_DS60_DR0.1_SFN32/inria/pred'
deeplab_base_dir = r'/hdd/Results/mmd/UnetCrop_inria_mmd_loo_5050_{}_1_PS(572, 572)_BS5_' \
                   r'EP40_LR1e-05_DS30_DR0.1_SFN32/inria/pred'
force_run = True

result_save_dir = list()
result_save_dir.append(os.path.join(img_dir, 'unet_deeplab_fn_building_large', 'both'))
result_save_dir.append(os.path.join(img_dir, 'unet_deeplab_fn_building_large', 'loo'))
result_save_dir.append(os.path.join(img_dir, 'unet_deeplab_fn_building_large', 'mmd'))
for result_dir in result_save_dir:
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

for city_num in [0, 1]:
    both_miss_size = []
    unet_miss_size = []
    deeplab_miss_size = []
    if force_run:
        for val_img_cnt in range(1, 6):
            rgb = imageio.imread(os.path.join(truth_dir, '{}{}_RGB.tif'.format(city_list[city_num], val_img_cnt)))

            truth_img_name = os.path.join(truth_dir, '{}{}_GT.tif'.format(city_list[city_num], val_img_cnt))
            unet_base_img_name = os.path.join(cnn_base_dir.format(city_num), '{}{}.png'.format(city_list[city_num], val_img_cnt))
            deeplab_base_img_name = os.path.join(deeplab_base_dir.format(city_num), '{}{}.png'.format(city_list[city_num], val_img_cnt))

            truth, unet_base, deeplab_base = imageio.imread(truth_img_name), imageio.imread(unet_base_img_name), \
                                             imageio.imread(deeplab_base_img_name)

            truth = truth / 255
            #unet_base = unet_base / 255
            #deeplab_base = deeplab_base / 255

            assert np.all(np.unique(truth) == np.array([0, 1])) and np.all(np.unique(unet_base) == np.array([0, 1])) \
                   and np.all(np.unique(deeplab_base) == np.array([0, 1]))

            for cnt, (fig, flag, bs) in enumerate(visualize_fn_object(rgb, truth, unet_base, deeplab_base)):
                plt.savefig(os.path.join(result_save_dir[flag-1], '{}{}_{}.png'.format(city_list[city_num],
                                                                                       val_img_cnt, cnt)))
                plt.close(fig)
                if flag == 1:
                    both_miss_size.append(bs)
                elif flag == 2:
                    unet_miss_size.append(bs)
                elif flag == 3:
                    deeplab_miss_size.append(bs)
        both_miss_size = np.array(both_miss_size)
        unet_miss_size = np.array(unet_miss_size)
        deeplab_miss_size = np.array(deeplab_miss_size)
        np.save(os.path.join(task_dir, '{}_both.npy'.format(city_list[city_num])), both_miss_size)
        np.save(os.path.join(task_dir, '{}_loo.npy'.format(city_list[city_num])), unet_miss_size)
        np.save(os.path.join(task_dir, '{}_mmd.npy'.format(city_list[city_num])), deeplab_miss_size)
    else:
        both_miss_size = np.load(os.path.join(task_dir, '{}_both.npy'.format(city_list[city_num])))
        unet_miss_size = np.load(os.path.join(task_dir, '{}_loo.npy'.format(city_list[city_num])))
        deeplab_miss_size = np.load(os.path.join(task_dir, '{}_mmd.npy'.format(city_list[city_num])))

    colors = util_functions.get_default_colors()
    fig = plt.figure(figsize=(14, 6))
    grid = plt.GridSpec(2, 2, wspace=0.15, hspace=0.3)
    ax1 = plt.subplot(grid[:, 0])
    plt.hist(both_miss_size, bins=range(0, 8080, 80), label='Both')
    plt.hist(unet_miss_size, bins=range(0, 8080, 80), label='LOO')
    plt.hist(deeplab_miss_size, bins=range(0, 8080, 80), label='MMD')
    plt.legend()
    plt.xlabel('Size of the Building')
    plt.ylabel('CNT')
    plt.title('Size of the Buidings Missed in {}'.format(city_list[city_num].title()))
    x1, x2, y1, y2 = -30, 2000, 200, 200
    ax2 = plt.subplot(grid[0, 1], sharex=ax1, sharey=ax1)
    plt.hist(unet_miss_size, bins=range(0, 8080, 80), label='LOO', facecolor=colors[1])
    plt.axvline(x=1000, color='r', linestyle='--')
    plt.text(x1, y1, 'FN={}'.format(np.sum(unet_miss_size < 1000)))
    plt.text(x2, y2, 'FN={}'.format(np.sum(unet_miss_size >= 1000)))
    plt.title('LOO FN={}'.format(int(len(unet_miss_size))))
    plt.ylabel('CNT')
    ax3 = plt.subplot(grid[1, 1], sharex=ax1, sharey=ax1)
    plt.hist(deeplab_miss_size, bins=range(0, 8080, 80), label='MMD', facecolor=colors[2])
    plt.axvline(x=1000, color='r', linestyle='--')
    plt.text(x1, y1, 'FN={}'.format(np.sum(deeplab_miss_size < 1000)))
    plt.text(x2, y2, 'FN={}'.format(np.sum(deeplab_miss_size >= 1000)))
    plt.title('MMD FN={}'.format(int(len(deeplab_miss_size))))
    plt.xlabel('Size of the Building')
    plt.ylabel('CNT')
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, 'miss_building_size_{}_large.png'.format(city_list[city_num])), bbox_inches='tight')
    plt.show()
