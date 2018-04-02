import os
import imageio
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
import utils


def get_error_vs_dist(model):
    patch_dir = '/hdd/Temp/IGARSS2018'
    if model == 'UnetCrop2':
        size = 388
    elif model == 'Deeplab321':
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
            r = i - size/2
            if r not in error_dict:
                error_dict[r] = 0

            error_dict[r] += error_map[i][j]

    sort_dist = np.sort(list(error_dict.keys()))
    for dist in sort_dist:
        dist_array.append(dist)
        error_array.append(error_dict[dist])
    #error_array = error_array/np.max(error_array)
    return dist_array, error_array


img_dir, task_dir = utils.get_task_img_folder()

nocrop_dist, nocrop_error = get_error_vs_dist('UnetNoCrop')
crop_dist, crop_error = get_error_vs_dist('Deeplab321')
nocrop_error = nocrop_error/np.sum(nocrop_error) * 100
crop_error = crop_error/np.sum(crop_error) * 100

matplotlib.rcParams.update({'font.size': 14})
plt.figure(figsize=(8, 4))
#plt.subplot(211)
#plt.subplot2grid((6, 1), (0, 0), rowspan=3)
plt.plot(np.array(nocrop_dist), np.array(nocrop_error), label='U-Net Zero-padding')
plt.plot(np.array(crop_dist), np.array(crop_error), label='DeepLab-CRF')
plt.legend(loc='center right')
plt.xlabel('Horizontal Dist to Center')
plt.grid('on')
#plt.text(-300, 20, '(a)')
plt.ylabel('%Errors')

#plt.subplot(212)
'''plt.subplot2grid((6, 1), (4, 0), rowspan=2)
ind = np.arange(2)
run_time = np.array([732.06, 624.62])
plt.bar(ind[0], run_time[0], 0.2)
plt.bar(ind[1], run_time[1], 0.2)
plt.xticks(ind, ['Zero Padding', 'No Zero Padding'])
plt.xlim([-0.5, 1.5])
plt.ylim([500, 800])
plt.xlabel('')
plt.ylabel('Time:s')
plt.text(-0.45, 600, '(b)')'''
plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'error_vs_dist_unet2.png'))
plt.show()
