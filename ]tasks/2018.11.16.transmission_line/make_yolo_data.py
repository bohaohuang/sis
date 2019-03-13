"""
Read all files in transmission line dataset, extract them into patches, add bounding boxes data to the patch, save them
into tf record file

Note:
This file has been modified so that it only predicts class as T
"""

IS_DCC = True
CITY_NAME = 'Wilmington'


import os
import io
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from glob import glob

import sys
if not IS_DCC:
    sys.path.append(r'/home/lab/Documents/bohao/code/third_party/models/research')
else:
    sys.path.append(r'/dscrhome/bh163/code/models/research')
    sys.path.append(r'/dscrhome/bh163/code/ersa')

from object_detection.utils import dataset_util

from natsort import natsorted
from skimage.draw import polygon
import ersa_utils

PATCH_SIZE = (500, 500)
ENCODER = {'DT': 0, 'TT': 1, 'T': 0}


def check_bounds(cc, rr, size_x, size_y):
    cc = np.maximum(np.minimum(cc, size_x), 0)
    rr = np.maximum(np.minimum(rr, size_y), 0)
    return cc,rr


def extract_grids(img, patch_size_h, patch_size_w):
    """
    Get patch grids for given image
    :param img:
    :param patch_size_h:
    :param patch_size_w:
    :return:
    """
    h, w, _ = img.shape
    h_cells = int(np.ceil(h / patch_size_h))
    w_cells = int(np.ceil(w / patch_size_w))
    if h % patch_size_h == 0:
        h_steps = np.arange(0, h, patch_size_h).astype(int)
    else:
        h_steps = np.append(np.arange(0, h-patch_size_h, patch_size_h).astype(int), h-patch_size_h)
    if w % patch_size_w == 0:
        w_steps = np.arange(0, w, patch_size_w).astype(int)
    else:
        w_steps = np.append(np.arange(0, w-patch_size_w, patch_size_w).astype(int), w-patch_size_w)
    grid_cell = [[{} for _ in range(w_cells)] for _ in range(h_cells)]
    for i in range(w_cells):
        for j in range(h_cells):
            grid_cell[j][i]['h'] = h_steps[j]
            grid_cell[j][i]['w'] = w_steps[i]
            grid_cell[j][i]['label'] = []
            grid_cell[j][i]['box'] = []
    return grid_cell, h_steps, w_steps


def get_cell_id(y, x, h_steps, w_steps):
    h_id_0, w_id_0 = None, None
    for cnt, hs in enumerate(h_steps):
        if hs <= y[0] < hs + PATCH_SIZE[0]:
            h_id_0 = cnt
            break
    for cnt, ws in enumerate(w_steps):
        if ws <= x[0] < ws + PATCH_SIZE[1]:
            w_id_0 = cnt
    return h_id_0, w_id_0


def read_polygon_csv_data(csv_file):
    label_order = ['SS', 'OT', 'DT', 'TT', 'OL', 'DL', 'TL']
    df = pd.read_csv(csv_file)
    df['temp_label'] = pd.Categorical(df['Label'], categories=label_order, ordered=True)
    df.sort_values('temp_label', inplace=True, kind='mergesort')

    for name, group in df.groupby('Object', sort=False):
        label = group['Label'].values[0]
        if group['Type'].values[0] == 'Polygon' and label in ENCODER:
            x, y = polygon(group['X'].values, group['Y'].values)
            y, x = check_bounds(x, y, df['width'][0] - 1, df['height'][0] - 1)
            yield label, x, y


def get_bounding_box(y, x):
    y_min = np.min(y).astype(int)
    x_min = np.min(x).astype(int)
    y_max = np.max(y).astype(int)
    x_max = np.max(x).astype(int)
    return max(y_min, 0), max(x_min, 0), min(y_max, PATCH_SIZE[0]), min(x_max, PATCH_SIZE[0])


def write_data_info(rgb_files, csv_files, save_dir):
    for rgb_file, csv_file in zip(rgb_files, csv_files):
        print('Processing data {}'.format(os.path.basename(rgb_file)[:-3]))
        save_name = os.path.basename(rgb_file)[:-4] + '.npy'
        rgb = ersa_utils.load_file(rgb_file)
        coords, h_steps, w_steps = extract_grids(rgb, PATCH_SIZE[0], PATCH_SIZE[1])

        for label, y, x in read_polygon_csv_data(csv_file):
            h_id_0, w_id_0 = get_cell_id(y, x, h_steps, w_steps)
            h_start = coords[h_id_0][w_id_0]['h']
            w_start = coords[h_id_0][w_id_0]['w']
            box = get_bounding_box(y-h_start, x-w_start)

            # FIXME only label them as T
            # coords[h_id_0][w_id_0]['label'].append(label)
            coords[h_id_0][w_id_0]['label'].append('T')
            coords[h_id_0][w_id_0]['box'].append(box)
        ersa_utils.save_file(os.path.join(save_dir, save_name), coords)


def make_dataset(rgb_files, info_dir, store_dir, city_name=''):
    writer_train = open(os.path.join(store_dir, 'data', 'train_{}_T.txt'.format(city_name)), 'w+')
    writer_valid = open(os.path.join(store_dir, 'data', 'test_{}_T.txt'.format(city_name)), 'w+')

    for rgb_file_name in rgb_files:
        file_name = os.path.basename(rgb_file_name[:-4])
        city_id = int(file_name.split('_')[-1])
        if city_id <= 3:
            print('Processing file {} in validation set'.format(file_name))
            is_val = True
        else:
            print('Processing file {} in training set'.format(file_name))
            is_val = False

        rgb = ersa_utils.load_file(rgb_file_name)
        npy_file_name = os.path.join(info_dir, os.path.basename(rgb_file_name[:-4] + '.npy'))
        coords = ersa_utils.load_file(npy_file_name)

        patch_cnt = 0
        for line in coords:
            for cell in line:
                patch_cnt += 1
                patch_file_name = os.path.basename(rgb_file_name[:-4] + '_{}.jpg'.format(patch_cnt))
                img_name = os.path.join(store_dir, 'build/darknet/x64/data/obj', patch_file_name)
                lbl_name = os.path.join(store_dir, 'build/darknet/x64/data/obj',
                                         os.path.basename(rgb_file_name[:-4] + '_{}.txt'.format(patch_cnt)))
                img = rgb[cell['h']:cell['h']+PATCH_SIZE[0], cell['w']:cell['w']+PATCH_SIZE[1], :]
                label = cell['label']
                # assert np.unique(label) == ['DT'] or label == []
                box = cell['box']
                ersa_utils.save_file(img_name, img)
                write_lbl(lbl_name, label, box, PATCH_SIZE)
                if is_val:
                    writer_valid.write('{}\n'.format(img_name))
                else:
                    writer_train.write('{}\n'.format(img_name))

    writer_train.close()
    writer_valid.close()


def write_lbl(lbl_name, label, box, patch_size):
    height, width = patch_size
    with open(lbl_name, 'w+') as f:
        for lbl, bb in zip(label, box):
            x = bb[1] / width
            y = bb[0] / height
            w = bb[3] / width - x
            h = bb[2] / height - y
            c = ENCODER[lbl]
            f.write('{} {} {} {} {}'.format(c, x, y, w, h))


if __name__ == '__main__':
    # settings
    if not IS_DCC:
        data_dir = r'/home/lab/Documents/bohao/data/transmission_line'
        yolo_dir = r'/home/lab/Documents/bohao/code/third_party/darknet'
        save_dir = os.path.join(data_dir, 'info')
        store_dir = os.path.join(data_dir, 'patches')
        tf_dir = os.path.join(data_dir, 'data')
        ersa_utils.make_dir_if_not_exist(save_dir)
        ersa_utils.make_dir_if_not_exist(store_dir)
        ersa_utils.make_dir_if_not_exist(tf_dir)
    else:
        data_dir = r'/work/bh163/misc/object_detection'
        yolo_dir = r'/dscrhome/bh163/code/darknet'
        save_dir = os.path.join(data_dir, 'info')
        store_dir = os.path.join(data_dir, 'patches')
        tf_dir = os.path.join(data_dir, 'data')
        ersa_utils.make_dir_if_not_exist(save_dir)
        ersa_utils.make_dir_if_not_exist(store_dir)
        ersa_utils.make_dir_if_not_exist(tf_dir)

    # get files
    rgb_files = natsorted([a for a in glob(os.path.join(data_dir, 'raw', '*.tif'))
                           if 'multiclass' not in a and CITY_NAME in a])
    csv_files = natsorted([a for a in glob(os.path.join(data_dir, 'raw', '*.csv')) if CITY_NAME in a])
    write_data_info(rgb_files, csv_files, save_dir)
    make_dataset(rgb_files, save_dir, yolo_dir, city_name=CITY_NAME)
