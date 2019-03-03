import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from glob import glob
from tqdm import tqdm
from scipy import ndimage
from skimage import measure
import sis_utils
import ersa_utils


def increase_size(x1, x2, window_size):
    if x2 - x1 < window_size:
        x1 = max(0, x1 - window_size//2)
        x2 = x1 + window_size
    return x1, x2


def make_smallest_region(loc, window_size=300, num=3):
    x1, x2, y1, y2 = loc[0].start, loc[0].stop, loc[1].start, loc[1].stop
    x1_new, x2_new = increase_size(x1, x2, window_size)
    y1_new, y2_new = increase_size(y1, y2, window_size)

    rect = []
    for i in range(num):
        rect.append(patches.Rectangle((y1-y1_new, x1-x1_new), y2-y1, x2-x1, linewidth=1, edgecolor='r',
                                      facecolor='none'))
    return [slice(x1_new, x2_new), slice(y1_new, y2_new)], rect


def make_cmp_plot(rgb, truth, prefix, rect, object_size):
    fig = plt.figure(figsize=(6, 3.5))
    ax1 = plt.subplot(121)
    plt.imshow(rgb)
    ax1.add_patch(rect[0])
    plt.title('{} Size={}'.format(prefix.title(), int(object_size)))
    plt.axis('off')
    plt.subplot(122)
    plt.imshow(truth, vmin=0, vmax=7)
    plt.axis('off')
    plt.tight_layout()

    return fig


def get_object_patch(rgb, gt, gt2show, prefix):
    h, w, _ = rgb.shape
    lbl = measure.label(gt)
    object_idx = np.unique(lbl)

    for cnt, idx in enumerate(tqdm(object_idx[1:])):
        object_size = np.sum(gt[np.where(lbl == idx)])
        temp_mask = np.zeros((h, w), dtype=np.uint8)
        temp_mask[np.where(lbl == idx)] = 1
        loc = ndimage.find_objects(temp_mask)[0]
        loc, rect = make_smallest_region(loc)

        fig = make_cmp_plot(rgb[loc], gt2show[loc], '{}_{}'.format(prefix, cnt), rect, object_size)
        yield fig, object_size, cnt


data_dir = r'/home/lab/Documents/bohao/data/transmission_line'
city_list = ['Tucson', 'Colwich', 'Clyde', 'Wilmington']
img_dir, task_dir = sis_utils.get_task_img_folder()
tower_vals = [3, 1, 5]
tower_names = ['Transmission_Tower', 'Distribution_Tower', 'Other_Tower']

tower_dict = {}
for cn in city_list:
    tower_type = {}
    for tt in tower_names:
        tower_type[tt] = [0]
    tower_dict[cn] = tower_type

for tn in tower_names:
    ersa_utils.make_dir_if_not_exist(os.path.join(img_dir, tn))

for city_id in range(len(city_list)):
    gt_files = sorted(glob(os.path.join(data_dir, '*{}*_multiclass.tif'.format(city_list[city_id]))))
    rgb_files = ['_'.join(a.split('_')[:-1])+'.tif' for a in gt_files]
    for rgb_file_name, gt_file_name in zip(rgb_files, gt_files):
        rgb = ersa_utils.load_file(rgb_file_name)
        gt = ersa_utils.load_file(gt_file_name)
        prefix = os.path.basename(rgb_file_name)[7:-4]

        for tower_val, tower_name in zip(tower_vals, tower_names):
            if tower_val in np.unique(gt):
                gt2show = np.copy(gt)
                gt2cmp = np.copy(gt)
                gt2cmp = (gt == tower_val).astype(np.uint8)
                for fig, object_size, cnt in get_object_patch(rgb, gt2cmp, gt2show, prefix):
                    save_name = os.path.join(img_dir, tower_name, '{}_{}_{}.png'.format(prefix, tower_name, cnt))
                    plt.savefig(save_name)
                    plt.close(fig)

                    tower_dict[city_list[city_id]][tower_name].append(object_size)

dict_save_name = os.path.join(task_dir, 'tower_dict.pkl')
ersa_utils.save_file(dict_save_name, tower_dict)
