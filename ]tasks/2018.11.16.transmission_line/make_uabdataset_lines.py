import os
import cv2
import numpy as np
import pandas as pd
from glob import glob
from natsort import natsorted
from skimage.draw import line, polygon
import ersa_utils


# Settings
ENCODER = {'DL': 1, 'TL': 1, 'DT': 1, 'TT': 1, 'L': 1, 'T': 1}


def check_bounds(cc, rr, size_x, size_y):
    cc = np.maximum(np.minimum(cc, size_x), 0)
    rr = np.maximum(np.minimum(rr, size_y), 0)
    return cc, rr


def read_polygon_csv_data(csv_file):
    label_order = ['SS', 'OT', 'DT', 'TT', 'OL', 'DL', 'TL', 'T', 'L']
    df = pd.read_csv(csv_file)
    df['temp_label'] = pd.Categorical(df['Label'], categories=label_order, ordered=True)
    df.sort_values('temp_label', inplace=True, kind='mergesort')

    for name, group in df.groupby('Object', sort=False):
        label = group['Label'].values[0]
        if group['Type'].values[0] == 'Line' and label in ENCODER:
            for j in range(group.shape[0] - 1):
                r0, c0 = int(group['X'].values[j]), int(group['Y'].values[j])
                r1, c1 = int(group['X'].values[j + 1]), int(group['Y'].values[j + 1])
                cc, rr = line(r0, c0, r1, c1)
                cc, rr = check_bounds(cc, rr, df['width'][0] - 1, df['height'][0] - 1)
                yield label, cc, rr


def read_polygon_csv_data_towers(csv_file):
    label_order = ['SS', 'OT', 'DT', 'TT', 'OL', 'DL', 'TL', 'T', 'L']
    df = pd.read_csv(csv_file)
    df['temp_label'] = pd.Categorical(df['Label'], categories=label_order, ordered=True)
    df.sort_values('temp_label', inplace=True, kind='mergesort')

    for name, group in df.groupby('Object', sort=False):
        label = group['Label'].values[0]
        if group['Type'].values[0] == 'Polygon' and label in ENCODER:
            x, y = polygon(group['X'].values, group['Y'].values)
            y, x = check_bounds(x, y, df['width'][0] - 1, df['height'][0] - 1)
            yield label, x, y


def write_data_info(rgb_files, csv_files, save_dir):
    for rgb_file, csv_file in zip(rgb_files, csv_files):
        print('Processing data {} ...'.format(os.path.basename(rgb_file)[:-3]), end='')
        if 'NZ' in rgb_file and 'resize' in rgb_file:
            city_name = os.path.basename(rgb_file[:-4]).split('_')[1]
            if 'Palmerston North' in city_name:
                city_name = 'PalmerstonNorth'
            city_id = os.path.basename(rgb_file[:-4]).split('_')[-2]
        else:
            city_name = os.path.basename(rgb_file[:-4]).split('_')[2]
            city_id = os.path.basename(rgb_file[:-4]).split('_')[-1]

        rgb_save_name = '{}{}_RGB.tif'.format(city_name, city_id)
        gt_save_name = '{}{}_GT.png'.format(city_name, city_id)
        tw_save_name = '{}{}_TW.png'.format(city_name, city_id)

        rgb = ersa_utils.load_file(rgb_file)
        h, w = rgb.shape[:2]
        gt = np.zeros((h, w), dtype=np.uint8)
        gt_towers = np.zeros((h, w), dtype=np.uint8)

        for label, y, x in read_polygon_csv_data(csv_file):
            try:
                gt[x, y] = ENCODER[label]
            except IndexError:
                pass

        for label, y, x in read_polygon_csv_data_towers(csv_file):
            try:
                gt_towers[y, x] = ENCODER[label]
            except IndexError:
                pass

        # dilation
        kernel = np.ones((15, 15), np.uint8)
        gt = cv2.dilate(gt, kernel, iterations=1)

        '''from visualize import visualize_utils
        visualize_utils.compare_figures([rgb, gt, gt_towers], (1, 3), fig_size=(15, 5))'''

        ersa_utils.save_file(os.path.join(save_dir, rgb_save_name), rgb[:, :, :3])
        ersa_utils.save_file(os.path.join(save_dir, gt_save_name), gt)
        ersa_utils.save_file(os.path.join(save_dir, tw_save_name), gt_towers)

        print('Done!')


data_dir = r'/home/lab/Documents/bohao/data/transmission_line/raw2'
# get files
rgb_files = natsorted([a for a in glob(os.path.join(data_dir, '*.tif'))
                       if 'multiclass' not in a])
# csv_files = natsorted(glob(os.path.join(data_dir, '*.csv')))
csv_files_temp = natsorted([a for a in glob(os.path.join(data_dir, '*.csv'))])
csv_files = []
for c in csv_files_temp:
    if 'NZ' in c:
        if 'resize' in c:
            csv_files.append(c)
    else:
        csv_files.append(c)
save_dir = r'/media/ei-edl01/data/uab_datasets/lines_v3/data/Original_Tiles'
ersa_utils.make_dir_if_not_exist(save_dir)

write_data_info(rgb_files, csv_files, save_dir)
