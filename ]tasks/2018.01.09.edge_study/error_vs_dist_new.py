import os
import imageio
import scipy.signal
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import utils


def exponential_smoothing(series, alpha=1):
    result = [series[0]] # first value is same as series
    for n in range(1, series.shape[0]):
        result.append(alpha * series[n] + (1 - alpha) * result[n-1])
    return np.array(result)


def get_error_vs_dist(error_map, size):
    # compute error
    error_dict = {}
    dist_array = []
    error_array = []
    for i in range(size):
        for j in range(size):
            r = max((abs(i - size/2), abs(j - size/2)))
            if r not in error_dict:
                error_dict[r] = 0

            error_dict[r] += error_map[i][j]

    sort_dist = np.sort(list(error_dict.keys()))
    for dist in sort_dist:
        dist_array.append(dist)
        error_array.append(error_dict[dist])
    return dist_array, error_array


def get_error_vs_dist2(error_map, size):
    # compute error
    dist_array = []
    error_array = []
    for i in range(int(size/2)):
        error_array.append(0)
        error_array[-1] += np.sum(error_map[i:size-1-i, i])
        error_array[-1] += np.sum(error_map[i, i:size-1-i])
        error_array[-1] += np.sum(error_map[i:size-1-i, size-1-i])
        error_array[-1] += np.sum(error_map[size-1-i, i:size-1-i])
        error_array[-1] -= (error_map[i, i] + error_map[i, size-1-i] +
                            error_map[size-1-i, i] + error_map[size-1-i, size-1-i])
        error_array[-1] /= (size - 2*i)*4 - 4
        dist_array.append(size/2 - i)
    return dist_array[::-1], error_array[::-1]


parent_dir = r'/hdd/Temp/patches'
input_sizes = [572, 828, 1084, 1340, 1596, 1852, 2092, 2332, 2636]
error_cnt = []
img_dir, task_dir = utils.get_task_img_folder()

for size in tqdm(input_sizes):
    save_file = os.path.join(task_dir, 'error_array_{}_lmax2.npy'.format(size))

    if not os.path.exists(save_file):
        img_dir = os.path.join(parent_dir, '{}'.format(size))
        error_map = np.zeros((size-184, size-184))
        for city in ['austin', 'chicago', 'kitsap', 'tyrol-w', 'vienna']:
            for tile_id in range(5):
                for cnt in range(196):
                    img_name = '{}{}_{}.png'.format(city, tile_id+1, cnt+1)
                    error_map += imageio.imread(os.path.join(img_dir, img_name))/255
        dist_array, error_array = get_error_vs_dist2(error_map/196/25, size-184)
        # print(dist_array)
        np.save(save_file, [dist_array, error_array])
    else:
        dist_array, error_array = np.load(save_file)

    #plt.subplot(121)
    print(error_array[:10])
    plt.plot(np.array(dist_array), scipy.signal.medfilt(np.array(error_array), [51]),
             label='{}:{}'.format(size, np.sum(error_array)))
    #plt.subplot(122)
    #plt.plot(np.array(dist_array)[:], (np.array(error_array))[:], label='{}:{}'.format(size, np.sum(error_array)))

plt.ylim(0.030, 0.040)
plt.tight_layout()
plt.show()
