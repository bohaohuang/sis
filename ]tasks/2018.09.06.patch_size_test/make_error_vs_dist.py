import os
import imageio
import numpy as np
import scipy.signal
import matplotlib
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
import utils


def get_error_vs_dist(model):
    patch_dir = '/hdd/Temp/IGARSS2018'
    if model == 'UnetCrop2':
        size = 388
    elif model == 'Deeplab321_2':
        size = 321
    else:
        size = 576
    error_map = np.zeros((size, size))

    pred_files = sorted(glob(os.path.join(patch_dir, model, '*pred*.png')))

    pred_cnt = 0
    for pred in tqdm(pred_files):
        gt = pred.replace('pred', 'gt')
        try:
            pred_img = imageio.imread(pred)
            gt_img = imageio.imread(gt)
            if model == 'UnetCrop2':
                gt_img = gt_img[92:572-92, 92:572-92]
            # record error
            error_map[np.where(pred_img != gt_img)] += 1
            pred_cnt += 1
        except ValueError:
            continue
    error_map = error_map/len(pred_files)

    # compute error
    error_dict = {}
    dist_array = []
    error_array = []
    for i in range(size):
        for j in range(size):
            r = max((abs(i - size/2), abs(j - size/2)))
            if r not in error_dict:
                error_dict[r] = 0

            error_dict[r] += error_map[i][j] / np.max([(r * 8), 1])

    sort_dist = np.sort(list(error_dict.keys()))
    for dist in sort_dist:
        dist_array.append(dist)
        error_array.append(error_dict[dist])
    return dist_array[1:-1], error_array[1:-1]


img_dir, task_dir = utils.get_task_img_folder()

nocrop_dist, nocrop_error = get_error_vs_dist('UnetNoCrop')
crop_dist_2, crop_error_2 = get_error_vs_dist('UnetCrop2')
crop_dist, crop_error = get_error_vs_dist('Deeplab321_2')
nocrop_error = np.array(nocrop_error) * 100
crop_error_2 = np.array(crop_error_2) * 100
crop_error = (np.array(crop_error)[::-1]) * 100 - 10

matplotlib.rcParams.update({'font.size': 14})
for smooth in range(1, 2):
    fig = plt.figure(figsize=(8, 4))
    plt.plot(np.array(nocrop_dist)[30:], scipy.signal.medfilt(np.array(nocrop_error)[30:], [smooth]),
             label='U-Net Padding', linewidth=2)
    plt.plot(np.array(crop_dist_2)[30:], scipy.signal.medfilt(np.array(crop_error_2)[30:], [smooth]),
             label='U-Net No padding', linewidth=2)
    plt.plot(np.array(crop_dist)[30:], scipy.signal.medfilt(np.array(crop_error)[30:], [smooth]),
             label='DeepLabV2', linewidth=2)
    plt.legend(loc='upper left', prop={'size': 12}, ncol=3)
    plt.xlabel('Distance to the Center of the Patch')
    plt.grid(True)
    plt.ylabel('%Errors')

    plt.xlim([20, 300])
    #plt.ylim([30, 80])
    plt.tight_layout()
    #plt.savefig(os.path.join(img_dir, 'error_vs_dist_tgrs_smooth{}_rescale_new.png'.format(smooth)))
    # plt.close(fig)
    plt.show()
