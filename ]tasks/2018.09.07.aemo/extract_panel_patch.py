import os
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


def make_subplot(patch, title_str):
    plt.imshow(patch)
    plt.title('{}'.format(title_str))
    plt.box(True)
    plt.xticks([])
    plt.yticks([])


def make_smallest_region(loc, window_size=200, num=3):
    x1, x2, y1, y2 = loc[0].start, loc[0].stop, loc[1].start, loc[1].stop
    x1_new, x2_new = increase_size(x1, x2, window_size)
    y1_new, y2_new = increase_size(y1, y2, window_size)

    rect = []
    for i in range(num):
        rect.append(patches.Rectangle((y1-y1_new, x1-x1_new), y2-y1, x2-x1, linewidth=2, edgecolor='r',
                                      facecolor='none'))
    return [slice(x1_new, x2_new), slice(y1_new, y2_new)], rect


def make_cmp_plot(rgb, truth, city_str, building_size):
    fig = plt.figure(figsize=(9, 3.5))
    ax1 = plt.subplot(121)
    plt.imshow(rgb)
    # ax1.add_patch(rect[0])
    plt.title('{} Size={}'.format(city_str.title(), int(building_size)))
    plt.axis('off')
    ax2 = plt.subplot(122)
    make_subplot(truth, 'GT')
    # ax2.add_patch(rect[1])
    plt.tight_layout()

    return fig


def visualize_fn_object(rgb, gt, title_str):
    h, w, _ = rgb.shape
    lbl = measure.label(gt)
    building_idx = np.unique(lbl)

    for cnt, idx in enumerate(tqdm(building_idx)):
        building_size = np.sum(gt[np.where(lbl == idx)])
        temp_mask = np.zeros((h, w), dtype=np.uint8)
        temp_mask[np.where(lbl == idx)] = 1
        loc = ndimage.find_objects(temp_mask)[0]
        loc, rect = make_smallest_region(loc)

        fig = make_cmp_plot(rgb[loc], gt[loc], '{}_{}'.format(title_str, cnt), building_size)
        yield fig, building_size

spca_dir = r'/media/ei-edl01/data/uab_datasets/spca/data/Original_Tiles'
aemo_dir = r'/home/lab/Documents/bohao/data/aemo/aemo_pad'
aemohist_dir = r'/hdd/ersa/preprocess/aemo_pad/hist_matching'
img_dir, task_dir = sis_utils.get_task_img_folder()

spca_files = glob(os.path.join(spca_dir, '*_RGB.jpg'))
idx = np.random.permutation(20)
spca_files = [spca_files[i] for i in idx]
aemo_files = glob(os.path.join(aemo_dir, '*rgb.tif'))
aemo_hist_files = glob(os.path.join(aemohist_dir, '*histRGB.tif'))
ersa_utils.make_dir_if_not_exist(os.path.join(img_dir, 'spca'))
ersa_utils.make_dir_if_not_exist(os.path.join(img_dir, 'aemo'))
ersa_utils.make_dir_if_not_exist(os.path.join(img_dir, 'aemohist'))

for spca_file in spca_files:
    title_str = os.path.basename(spca_file).split('_')[0]

    rgb = ersa_utils.load_file(spca_file)
    gt_file = spca_file[:-7] + 'GT.png'
    gt = ersa_utils.load_file(gt_file)

    for cnt, (fig, _) in enumerate(visualize_fn_object(rgb, gt, title_str)):
        plt.savefig(os.path.join(img_dir, 'spca', '{}_{}.png'.format(title_str, cnt)))
        plt.close(fig)

for aemo_file in aemo_files:
    title_str = '_'.join(os.path.basename(aemo_file).split('_')[:2])

    rgb = ersa_utils.load_file(aemo_file)
    gt_file = aemo_file[:-7] + 'gt_d255.tif'
    gt = ersa_utils.load_file(gt_file)

    for cnt, (fig, _) in enumerate(visualize_fn_object(rgb, gt, title_str)):
        plt.savefig(os.path.join(img_dir, 'aemo', '{}_{}.png'.format(title_str, cnt)))
        plt.close(fig)

for aemo_file in aemo_hist_files:
    title_str = '_'.join(os.path.basename(aemo_file).split('_')[:2])

    rgb = ersa_utils.load_file(aemo_file)
    gt_file = os.path.join(aemo_dir, os.path.basename(aemo_file[:-15]) + 'gt_d255.tif')
    gt = ersa_utils.load_file(gt_file)

    for cnt, (fig, _) in enumerate(visualize_fn_object(rgb, gt, title_str)):
        plt.savefig(os.path.join(img_dir, 'aemohist', '{}_{}.png'.format(title_str, cnt)))
        plt.close(fig)
