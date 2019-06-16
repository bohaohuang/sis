"""

"""


# Built-in
import os
import json

# Libs
import numpy as np
import pandas as pd
from glob import glob
from natsort import natsorted
from skimage.draw import polygon

# Own modules
import ersa_utils
from visualize import visualize_utils

ENCODER = {'DT': 1, 'TT': 2, 'T': 1}


def read_json(file_name):
    with open(file_name) as f:
        data = json.load(f)
    return data


def dict_to_pd(d, height, width, factor=(1, 1), save_name=None):
    df = []
    objs = d['objects']
    typs = d['type']
    lbls = d['label']
    for cnt, (o, t, l) in enumerate(zip(objs, typs, lbls)):
        for w, h in o:
            df.append([l, cnt, t, w*factor[0], h*factor[1], height, width])
            if w*factor[0] >= width or h*factor[1] >= height:
                print(w*factor[0], h*factor[1], width, height)
    df = pd.DataFrame(df, columns=['Label', 'Object', 'Type', 'X', 'Y', 'height', 'width'])
    if save_name:
        df.to_csv(save_name)


def clean_artem():
    data_dir = r'/home/lab/Documents/bohao/data/transmission_line/raw2'
    csv_files = natsorted(glob(os.path.join(data_dir, '*.csv')))
    json_files = natsorted(glob(os.path.join(data_dir, '*.json')))
    csv_names = [os.path.splitext(a)[0] for a in csv_files]
    json_names = [os.path.splitext(a)[0] for a in json_files]

    for jn in json_names:
        if jn not in csv_names:
            d = read_json('{}.json'.format(jn))
            print(jn)
            if 'Dunedin' in jn:
                height = 4800
                width = 3200
                factor = (4800/5760, 3200/3840)
                save_name = '{}_resize.csv'.format(jn)
            elif 'Gisborne' in jn:
                height = 10020
                width = 5000
                factor = (10020/12040, 5000/6000)
                save_name = '{}_resize.csv'.format(jn)
            elif 'Colwich' in jn:
                height = 10000
                width = 10000
                factor = (1, 1)
                save_name = '{}.csv'.format(jn)
            else:
                raise NotImplementedError
            dict_to_pd(d, height, width, factor, save_name)


def clean_dataplus():
    data_dir = r'/home/lab/Documents/bohao/data/transmission_line/raw2'
    csv_files = natsorted(glob(os.path.join(data_dir, '*.csv')))
    factor = (4800/5760, 3200/3840)
    for cn in csv_files:
        if 'NZ' in cn and 'resize' not in cn:
            df = pd.read_csv(cn)
            save_name = '{}_resize.csv'.format(os.path.splitext(cn)[0])
            value = df.values.tolist()
            new_df = []
            for line in value:
                new_df.append([line[1], line[2], line[3], line[4]*factor[0], line[5]*factor[1], line[6], line[7]])
                if line[4]*factor[0] > 3200 or line[5]*factor[1] > 4800:
                    print(cn + '!!!!!!!!')
            print(cn)
            df = pd.DataFrame(new_df, columns=['Label', 'Object', 'Type', 'X', 'Y', 'height', 'width'])
            df.to_csv(save_name)


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


def check_annotations():
    data_dir = r'/home/lab/Documents/bohao/data/transmission_line/raw2'
    csv_files_temp = natsorted(glob(os.path.join(data_dir, '*.csv')))
    csv_files = []
    for c in csv_files_temp:
        if 'NZ' in c:
            if 'resize' in c:
                csv_files.append(c)
        else:
            csv_files.append(c)
    rgb_files = natsorted(glob(os.path.join(data_dir, '*.tif')))
    for cn, rn in zip(csv_files, rgb_files):
        if 'Dunedin' in cn:
            rgb = ersa_utils.load_file(rn)
            h, w = rgb.shape[:2]
            gt = np.zeros((h, w), dtype=np.uint8)
            for label, y, x in read_polygon_csv_data(cn):
                gt[y, x] = ENCODER[label]
            visualize_utils.compare_figures([rgb, gt], (1, 2), fig_size=(12, 5), show_axis=False)


if __name__ == '__main__':
    # clean_artem()
    clean_dataplus()
    # check_annotations()
