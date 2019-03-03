import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from skimage import measure
import sis_utils
import ersa_utils


def get_objects(gt):
    lbl = measure.label(gt)
    building_idx = np.unique(lbl)

    sizes = []
    coords = []

    for idx in building_idx:
        building_cord = np.where((lbl == idx))
        building_size = np.sum(gt[building_cord])
        sizes.append(building_size)
        coords.append(building_cord)

    return sizes, coords


def pred_stats(pred, gt):
    lbl = measure.label(gt)
    building_idx = np.unique(lbl)

    sizes = []
    accuracy = []

    for idx in building_idx:
        building_cord = np.where((lbl == idx))
        building_size = np.sum(gt[building_cord])
        if building_size > 0:
            building_acc = np.sum(pred[building_cord]) / building_size
            sizes.append(building_size)
            accuracy.append(building_acc)

    return sizes, accuracy


def plot_gt_panel_stats(gt_files):
    for gt_file in gt_files:
        tile_name = os.path.basename(gt_file)[:-9]
        gt = ersa_utils.load_file(gt_file)
        sizes, coords = get_objects(gt)
        plt.hist(sizes, bins=np.arange(2500))
        plt.xlabel('Panel Size')
        plt.ylabel('Cnts')
        plt.title(tile_name)
        plt.tight_layout()
        plt.savefig(os.path.join(img_dir, '{}_panel_size_stats.png'.format(tile_name)))
        plt.close()


def plot_pred_stats(gt_files, pred_files, model_name):
    size_all = []
    acc_all = []
    for gt_file, pred_file in zip(gt_files, pred_files):
        gt = ersa_utils.load_file(gt_file)
        pred = ersa_utils.load_file(pred_file)

        sizes, accuracy = pred_stats(pred, gt)
        size_all.append(sizes)
        acc_all.append(accuracy)
    plt.scatter(np.concatenate(size_all), np.concatenate(acc_all), s=8)
    plt.xlabel('Building Size')
    plt.ylabel('Accuracy')
    plt.title(model_name)
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, 'panel_size_vs_acc_{}.png'.format(model_name)))
    plt.close()


def create_pred_weight_map(gt_files, save_dir, thresh=200):
    for gt_file in gt_files:
        tile_name = os.path.basename(gt_file)[:-9]
        gt = ersa_utils.load_file(gt_file)
        lbl = measure.label(gt)
        building_idx = np.unique(lbl)
        gt_wm = np.zeros_like(gt, dtype=np.uint8)

        for idx in building_idx[1:]:
            building_cord = np.where((lbl == idx))
            building_size = np.sum(gt[building_cord])
            if building_size > thresh:
                gt_wm[building_cord] = 2
            else:
                gt_wm[building_cord] = 1
        ersa_utils.save_file(os.path.join(save_dir, '{}_wm.tif'.format(tile_name)), gt_wm)


if __name__ == '__main__':
    img_dir, task_dir = sis_utils.get_task_img_folder()
    data_dir = r'/home/lab/Documents/bohao/data/aemo/aemo_align'
    rgb_files = sorted(glob(os.path.join(data_dir, '*rgb.tif')))[-2:]
    gt_files = sorted(glob(os.path.join(data_dir, '*d255.tif')))[-2:]

    #create_pred_weight_map(gt_files, data_dir)

    model_dir = ['UnetCrop_aemo_reweight_0_PS(572, 572)_BS5_EP80_LR0.001_DS30_DR0.1_SFN32/aemo'
                 ]
    model_name = ['Reweight']
    for md, mn in zip(model_dir, model_name):
        pred_dir = r'/hdd/Results/aemo/uab/{}/pred'.format(md)
        pred_files = sorted(glob(os.path.join(pred_dir, '*.png')))
        plot_pred_stats(gt_files, pred_files, mn)
