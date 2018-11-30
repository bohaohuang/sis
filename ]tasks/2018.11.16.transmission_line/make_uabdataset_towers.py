import os
import numpy as np
import pandas as pd
from glob import glob
from natsort import natsorted
from skimage.draw import polygon
import ersa_utils


# Settings
ENCODER = {'DT': 1, 'TT': 2}


def check_bounds(cc, rr, size_x, size_y):
    cc = np.maximum(np.minimum(cc, size_x), 0)
    rr = np.maximum(np.minimum(rr, size_y), 0)
    return cc, rr


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


def write_data_info(rgb_files, csv_files, save_dir):
    for rgb_file, csv_file in zip(rgb_files, csv_files):
        print('Processing data {} ...'.format(os.path.basename(rgb_file)[:-3]), end='')
        city_name = os.path.basename(rgb_file[:-4]).split('_')[2]
        city_id = os.path.basename(rgb_file[:-4]).split('_')[-1]

        rgb_save_name = '{}{}_RGB.tif'.format(city_name, city_id)
        gt_save_name = '{}{}_GT.png'.format(city_name, city_id)

        rgb = ersa_utils.load_file(rgb_file)
        h, w = rgb.shape[:2]
        gt = np.zeros((h, w), dtype=np.uint8)

        for label, y, x in read_polygon_csv_data(csv_file):
            gt[y, x] = ENCODER[label]

        ersa_utils.save_file(os.path.join(save_dir, rgb_save_name), rgb)
        ersa_utils.save_file(os.path.join(save_dir, gt_save_name), gt)
        print('Done!')


data_dir = r'/home/lab/Documents/bohao/data/transmission_line/raw'
# get files
rgb_files = natsorted([a for a in glob(os.path.join(data_dir, '*.tif'))
                       if 'multiclass' not in a])
csv_files = natsorted(glob(os.path.join(data_dir, '*.csv')))
save_dir = r'/media/ei-edl01/data/uab_datasets/towers/data/Original_Tiles'

write_data_info(rgb_files, csv_files, save_dir)
