import os
import imageio
import scipy.signal
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm


def exponential_smoothing(series, alpha=1):
    result = [series[0]] # first value is same as series
    for n in range(1, series.shape[0]):
        result.append(alpha * series[n] + (1 - alpha) * result[n-1])
    return np.array(result)


def get_error_vs_dist(model, input_size):
    patch_dir = '/hdd/Temp/IGARSS2018'
    size = input_size - 184
    error_map = np.zeros((size, size))
    pixel_num = 0
    pred_files = sorted(glob(os.path.join(patch_dir, model, '*pred*.png')))

    pred_cnt = 0
    for pred in tqdm(pred_files):
        gt = pred.replace('pred', 'gt')
        try:
            pred_img = imageio.imread(pred)
            pixel_num += pred_img.shape[0] * pred_img.shape[1]
            gt_img = imageio.imread(gt)
            gt_img = gt_img[92:size+184-92, 92:size+184-92]
            # record error
            error_map[np.where(pred_img != gt_img)] += 1
            pred_cnt += 1
        except ValueError as e:
            print(e)
            continue
    #error_map = error_map/len(pred_files)

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
    return dist_array, error_array, pixel_num


matplotlib.rcParams.update({'font.size': 14})
error_cnt = []
pixel_cnt = []
plt.figure(figsize=(14, 6))
plt.subplot(121)
sizes = [572, 828, 1084, 1340, 1596, 1852, 2092, 2332, 2636]
for size in sizes: #[572, 828, 1084, 1340, 1596, 1852, 2092, 2332, 2636]:
    crop_dist, crop_error, pixel_num = get_error_vs_dist('UnetCrop{}'.format(size), size)
    #plt.plot(np.array(crop_dist), exponential_smoothing(np.array(crop_error)), label='{}:{}'.format(size, np.sum(crop_error)))
    plt.plot(np.array(crop_dist), scipy.signal.medfilt(np.array(crop_error), [9]),
             label='{}:{}'.format(size, np.sum(crop_error)))
    error_cnt.append(np.sum(crop_error)/pixel_num)
    pixel_cnt.append(pixel_num/(5000*5000*25))
plt.xlabel('Horizontal Dist to Center')
plt.ylabel('%Errors')
plt.legend(loc='best')

plt.subplot(122)
plt.plot(sizes, error_cnt)
#plt.plot(sizes, pixel_cnt)
#print(pixel_cnt)

plt.show()
