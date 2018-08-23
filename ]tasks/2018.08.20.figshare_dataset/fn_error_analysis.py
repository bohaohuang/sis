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


def make_cmp_plot(rgb, truth, base, pred_1, pred_2, city_str, rect, building_size, pred_tile, pred_title_cmp):
    pred_patch_base = make_error_image(truth, base)
    pred_patch_1 = make_error_image(truth, pred_1)
    pred_patch_2 = make_error_image(truth, pred_2)

    fig = plt.figure(figsize=(11, 3.5))
    ax1 = plt.subplot(141)
    plt.imshow(rgb)
    ax1.add_patch(rect[0])
    plt.title('{} Size={}'.format(city_str.title(), int(building_size)))
    plt.axis('off')
    ax2 = plt.subplot(142)
    make_subplot(pred_patch_base, 'Base')
    ax2.add_patch(rect[1])
    ax2 = plt.subplot(143)
    make_subplot(pred_patch_1, pred_tile)
    ax2.add_patch(rect[2])
    ax3 = plt.subplot(144)
    make_subplot(pred_patch_2, pred_title_cmp)
    ax3.add_patch(rect[3])
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


def visualize_fn_object(rgb, gt, base, pred_1, pred_2, pred_1_title, pred_2_title):
    h, w, _ = rgb.shape
    lbl = measure.label(gt)
    building_idx = np.unique(lbl)

    for cnt, idx in enumerate(tqdm(building_idx)):
        on_target_base = np.sum(base[np.where(lbl == idx)])
        on_target_pred_1 = np.sum(pred_1[np.where(lbl == idx)])
        on_target_pred_2 = np.sum(pred_2[np.where(lbl == idx)])
        building_size = np.sum(gt[np.where(lbl == idx)])
        flag = 0
        if on_target_base != 0 and on_target_pred_1 == 0 and on_target_pred_2 == 0:
            flag = 1
        elif on_target_base == 0 and on_target_pred_1 != 0 and on_target_pred_2 == 0:
            flag = 2
        elif on_target_base == 0 and on_target_pred_1 == 0 and on_target_pred_2 != 0:
            flag = 3
        elif on_target_base == 0 and on_target_pred_1 != 0 and on_target_pred_2 != 0:
            flag = 4
        if flag > 0:
            temp_mask = np.zeros((h, w), dtype=np.uint8)
            temp_mask[np.where(lbl == idx)] = 1
            loc = ndimage.find_objects(temp_mask)[0]
            loc, rect = make_smallest_region(loc, num=4)

            fig = make_cmp_plot(rgb[loc], gt[loc], base[loc], pred_1[loc], pred_2[loc],
                                '{}{}'.format(city_name, val_img_cnt), rect,
                                building_size, pred_1_title, pred_2_title)
            yield fig, flag, building_size


img_dir, task_dir = utils.get_task_img_folder()
city_name = 'atlanta'
truth_dir = r'/media/ei-edl01/data/uab_datasets/{}/data/Original_Tiles'.format(city_name)
base_dir = r'/hdd/Results/kyle/UnetCrop_inria_aug_grid_0_PS(572, 572)_BS5_' \
               r'EP100_LR0.0001_DS60_DR0.1_SFN32/{}/pred'.format(city_name)
mmd_base_dir = r'/hdd/Results/kyle/UnetCrop_inria_mmd_xregion_5050_atlanta_1_PS(572, 572)_BS5_' \
              r'EP40_LR1e-05_DS30_DR0.1_SFN32/{}/pred'.format(city_name)
dis_base_dir = r'/hdd/Results/kyle/UnetCrop_inria_distance_xregion_5050_atlanta_1_PS(572, 572)_BS5_' \
              r'EP40_LR1e-05_DS30_DR0.1_SFN32/{}/pred'.format(city_name)
force_run = True

result_save_dir = list()
result_save_dir.append(os.path.join(img_dir, 'mmd_dis_fn_building_large', 'miss'))
result_save_dir.append(os.path.join(img_dir, 'mmd_dis_fn_building_large', 'mmd'))
result_save_dir.append(os.path.join(img_dir, 'mmd_dis_fn_building_large', 'dis'))
result_save_dir.append(os.path.join(img_dir, 'mmd_dis_fn_building_large', 'both'))
for result_dir in result_save_dir:
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

both_miss_size = []
mmd_catch_size = []
dis_catch_size = []
if force_run:
    for val_img_cnt in range(1, 4):
        rgb = imageio.imread(os.path.join(truth_dir, '{}{}_RGB.tif'.format(city_name, val_img_cnt)))
        truth_img_name = os.path.join(truth_dir, '{}{}_GT.png'.format(city_name, val_img_cnt))
        base_img_name = os.path.join(base_dir, '{}{}.png'.format(city_name, val_img_cnt))
        mmd_base_img_name = os.path.join(mmd_base_dir, '{}{}.png'.format(city_name, val_img_cnt))
        dis_base_img_name = os.path.join(dis_base_dir, '{}{}.png'.format(city_name, val_img_cnt))

        truth, base, mmd, dis = imageio.imread(truth_img_name), imageio.imread(base_img_name), \
                                imageio.imread(mmd_base_img_name), imageio.imread(dis_base_img_name)

        assert np.all(np.unique(truth) == np.array([0, 1])) and np.all(np.unique(base) == np.array([0, 1])) \
               and np.all(np.unique(mmd) == np.array([0, 1])) and np.all(np.unique(dis) == np.array([0, 1]))

        for cnt, (fig, flag, bs) in enumerate(visualize_fn_object(rgb, truth, base, mmd, dis, 'MMD', 'DIS')):
            plt.savefig(os.path.join(result_save_dir[flag-1], '{}{}_{}.png'.format(city_name, val_img_cnt, cnt)))
            plt.close(fig)
            if flag == 1:
                both_miss_size.append(bs)
            elif flag == 2:
                mmd_catch_size.append(bs)
            elif flag == 3:
                dis_catch_size.append(bs)
    both_miss_size = np.array(both_miss_size)
    mmd_catch_size = np.array(mmd_catch_size)
    dis_catch_size = np.array(dis_catch_size)
    np.save(os.path.join(task_dir, '{}_both.npy'.format(city_name)), both_miss_size)
    np.save(os.path.join(task_dir, '{}_mmd.npy'.format(city_name)), mmd_catch_size)
    np.save(os.path.join(task_dir, '{}_dis.npy'.format(city_name)), dis_catch_size)
else:
    both_miss_size = np.load(os.path.join(task_dir, '{}_both.npy'.format(city_name)))
    mmd_catch_size = np.load(os.path.join(task_dir, '{}_mmd.npy'.format(city_name)))
    dis_catch_size = np.load(os.path.join(task_dir, '{}_dis.npy'.format(city_name)))

'''colors = util_functions.get_default_colors()
fig = plt.figure(figsize=(14, 6))
grid = plt.GridSpec(2, 2, wspace=0.15, hspace=0.3)
ax1 = plt.subplot(grid[:, 0])
plt.hist(both_miss_size, bins=range(0, 8080, 80), label='Both')
plt.hist(one_catch_size, bins=range(0, 8080, 80), label='U-Net')
plt.hist(deeplab_miss_size, bins=range(0, 8080, 80), label='DeepLab')
plt.legend()
plt.xlabel('Size of the Building')
plt.ylabel('CNT')
plt.title('Size of the Buidings Missed in {}'.format(city_list[city_num].title()))
x1, x2, y1, y2 = -30, 2000, 200, 200
ax2 = plt.subplot(grid[0, 1], sharex=ax1, sharey=ax1)
plt.hist(unet_miss_size, bins=range(0, 8080, 80), label='U-Net', facecolor=colors[1])
plt.axvline(x=1000, color='r', linestyle='--')
plt.text(x1, y1, 'FN={}'.format(np.sum(unet_miss_size < 1000)))
plt.text(x2, y2, 'FN={}'.format(np.sum(unet_miss_size >= 1000)))
plt.title('U-Net FN={}'.format(int(len(unet_miss_size))))
plt.ylabel('CNT')
ax3 = plt.subplot(grid[1, 1], sharex=ax1, sharey=ax1)
plt.hist(deeplab_miss_size, bins=range(0, 8080, 80), label='DeepLab', facecolor=colors[2])
plt.axvline(x=1000, color='r', linestyle='--')
plt.text(x1, y1, 'FN={}'.format(np.sum(deeplab_miss_size < 1000)))
plt.text(x2, y2, 'FN={}'.format(np.sum(deeplab_miss_size >= 1000)))
plt.title('DeepLab FN={}'.format(int(len(deeplab_miss_size))))
plt.xlabel('Size of the Building')
plt.ylabel('CNT')
plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'miss_building_size_{}_large.png'.format(city_list[city_num])), bbox_inches='tight')
plt.show()'''
