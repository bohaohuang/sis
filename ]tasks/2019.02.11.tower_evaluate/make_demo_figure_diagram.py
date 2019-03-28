"""

"""


# Built-in
import os

# Libs
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.ndimage.filters import gaussian_filter
from skimage import data, color, io, img_as_float

# Own modules
import sis_utils
from rst_utils import misc_utils
from post_processing_utils import load_data, add_points
from visualize import visualize_utils


def get_dirs():
    img_dir, task_dir = sis_utils.get_task_img_folder()
    dirs = {
        'task': task_dir,
        'image': img_dir,
        'raw': r'/home/lab/Documents/bohao/data/transmission_line/raw',
        'conf': r'/media/ei-edl01/user/bh163/tasks/2018.11.16.transmission_line/'
                r'confmap_uab_UnetCrop_lines_pw30_0_PS(572, 572)_BS5_EP100_LR0.0001_DS60_DR0.1_SFN32',
        'line': r'/media/ei-edl01/data/uab_datasets/lines/data/Original_Tiles'
    }
    return dirs


def get_pred_stats(pred):
    items = pred.strip().split(' ')
    obj_class = items[0]
    conf = float(items[1])
    x_min = int(items[2])
    y_min = int(items[3])
    x_max = int(items[4])
    y_max = int(items[5])
    return obj_class, conf, x_min, y_min, x_max, y_max


def tower_pred_demo(rgb, tower_pred, tower_gt, region):
    fig, ax = plt.subplots(1, figsize=(8, 8))
    ax.imshow(rgb)
    add_points(tower_gt, 'b', marker='s', size=80, alpha=1, edgecolor='k')

    for pred in tower_pred:
        obj_class, conf, x_min, y_min, x_max, y_max = get_pred_stats(pred)
        if conf > 0.5:
            rect = patches.Rectangle((x_min, y_min), x_max-x_min, y_max-y_min,
                                     linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

    plt.axis('off')
    if region:
        plt.xlim([region[2], region[3]])
        plt.ylim([region[1], region[0]])
    plt.tight_layout()


def orig_demo(rgb, region):
    fig, ax = plt.subplots(1, figsize=(8, 8))
    ax.imshow(rgb)
    plt.axis('off')
    if region:
        plt.xlim([region[2], region[3]])
        plt.ylim([region[1], region[0]])
    plt.tight_layout()


def lines_conf_demo(rgb, line_conf, region):
    if region:
        rgb = rgb[region[0]:region[1], region[2]:region[3]]
        line_conf = gaussian_filter(line_conf[region[0]:region[1], region[2]:region[3]], 25)
    else:
        rgb = rgb
        line_conf = gaussian_filter(line_conf, 25)
    '''line_zeros = np.zeros_like(line_conf)
    line_conf = np.dstack((line_conf, line_zeros, line_zeros))

    rgb_hsv = color.rgb2hsv(rgb)
    line_hsv = color.rgb2hsv(line_conf)
    rgb_hsv[..., 0] = line_hsv[..., 0]
    rgb_hsv[..., 1] = line_hsv[..., 1] * 0.2
    img = color.hsv2rgb(rgb_hsv)'''

    fig, ax = plt.subplots(1, figsize=(8, 8))
    ax.imshow(line_conf)
    plt.axis('off')
    plt.tight_layout()


def demo_plot(rgb, line_conf, pred_raw, tower_gt, region, dirs):
    rgb_orig = rgb
    if region:
        rgb = rgb[region[0]:region[1], region[2]:region[3]]
        line_conf = gaussian_filter(line_conf[region[0]:region[1], region[2]:region[3]], 5)
    else:
        rgb = rgb
        line_conf = gaussian_filter(line_conf, 5)

    #misc_utils.save_file(os.path.join(dirs['image'], 'demo_rgb.png'), rgb)
    #misc_utils.save_file(os.path.join(dirs['image'], 'demo_line.png'), line_conf)

    # visualize_utils.compare_figures([rgb, line_conf], (1, 2), show_axis=False, fig_size=(12, 5), show_fig=False)
    tower_pred_demo(rgb_orig, pred_raw, tower_gt, None)
    #plt.savefig(os.path.join(dirs['image'], 'demo_pred.png'))


if __name__ == '__main__':
    # settings
    city_id = 3
    tile_id = 2
    model_name = 'faster_rcnn'
    region = [800, 1400, 600, 1300]

    dirs = get_dirs()
    pred_raw, rgb, line_conf, line_gt, tower_gt, tower_pred, _ = load_data(dirs, model_name, city_id, tile_id)

    '''orig_demo(rgb, region)
    tower_pred_demo(rgb, pred_raw, tower_gt, region)
    lines_conf_demo(rgb, line_conf, region)
    plt.show()'''

    demo_plot(rgb, line_conf, pred_raw, tower_gt, region, dirs)
    plt.show()
