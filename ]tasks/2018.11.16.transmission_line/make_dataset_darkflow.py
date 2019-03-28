"""

"""


# Built-in
import os
import numpy as np
import pandas as pd
from glob import glob
from skimage.draw import polygon

# Libs
from natsort import natsorted

# Own modules
import ersa_utils


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


def get_cell_id(y, x, h_steps, w_steps, patch_size):
    h_id_0, w_id_0 = None, None
    for cnt, hs in enumerate(h_steps):
        if hs <= y[0] < hs + patch_size[0]:
            h_id_0 = cnt
            break
    for cnt, ws in enumerate(w_steps):
        if ws <= x[0] < ws + patch_size[1]:
            w_id_0 = cnt
    return h_id_0, w_id_0


def read_polygon_csv_data(csv_file, encoder):
    label_order = ['SS', 'OT', 'DT', 'TT', 'OL', 'DL', 'TL']
    df = pd.read_csv(csv_file)
    df['temp_label'] = pd.Categorical(df['Label'], categories=label_order, ordered=True)
    df.sort_values('temp_label', inplace=True, kind='mergesort')

    for name, group in df.groupby('Object', sort=False):
        label = group['Label'].values[0]
        if group['Type'].values[0] == 'Polygon' and label in encoder:
            x, y = polygon(group['X'].values, group['Y'].values)
            y, x = check_bounds(x, y, df['width'][0] - 1, df['height'][0] - 1)
            yield label, x, y


def get_bounding_box(y, x, patch_size):
    y_min = np.min(y).astype(int)
    x_min = np.min(x).astype(int)
    y_max = np.max(y).astype(int)
    x_max = np.max(x).astype(int)
    return max(y_min, 0), max(x_min, 0), min(y_max, patch_size[0]), min(x_max, patch_size[0])


def write_data_info(rgb_files, csv_files, save_dir, patch_size, encoder):
    for rgb_file, csv_file in zip(rgb_files, csv_files):
        print('Processing data {}'.format(os.path.basename(rgb_file)[:-3]))
        save_name = os.path.basename(rgb_file)[:-4] + '.npy'
        rgb = ersa_utils.load_file(rgb_file)
        coords, h_steps, w_steps = extract_grids(rgb, patch_size[0], patch_size[1])

        for label, y, x in read_polygon_csv_data(csv_file, encoder):
            h_id_0, w_id_0 = get_cell_id(y, x, h_steps, w_steps, patch_size)
            h_start = coords[h_id_0][w_id_0]['h']
            w_start = coords[h_id_0][w_id_0]['w']
            box = get_bounding_box(y-h_start, x-w_start, patch_size)

            coords[h_id_0][w_id_0]['label'].append(label)
            coords[h_id_0][w_id_0]['box'].append(box)
        ersa_utils.save_file(os.path.join(save_dir, save_name), coords)


def make_dataset(is_dcc, city_name, patch_size, encoder):
    # settings
    if not is_dcc:
        data_dir = r'/home/lab/Documents/bohao/data/transmission_line'
    else:
        data_dir = r'/work/bh163/misc/object_detection'
    base_dir = os.path.join(data_dir, 'yolo')
    save_dir = os.path.join(base_dir, 'temp')
    img_dir = os.path.join(base_dir, 'images', 'all')
    csv_dir = os.path.join(base_dir, 'csv')
    ersa_utils.make_dir_if_not_exist(img_dir)
    ersa_utils.make_dir_if_not_exist(csv_dir)
    ersa_utils.make_dir_if_not_exist(save_dir)
    ersa_utils.make_dir_if_not_exist(os.path.join(base_dir, 'images', 'train'))
    ersa_utils.make_dir_if_not_exist(os.path.join(base_dir, 'images', 'test'))
    ersa_utils.make_dir_if_not_exist(os.path.join(base_dir, 'labels', 'train'))
    ersa_utils.make_dir_if_not_exist(os.path.join(base_dir, 'labels', 'test'))

    # get files
    rgb_files = natsorted([a for a in glob(os.path.join(data_dir, 'raw', '*.tif'))
                           if 'multiclass' not in a and city_name in a])
    csv_files = natsorted([a for a in glob(os.path.join(data_dir, 'raw', '*.csv')) if city_name in a])
    write_data_info(rgb_files, csv_files, save_dir, patch_size, encoder)
    write_dataset(rgb_files, save_dir, img_dir, csv_dir, patch_size, city_name)


def write_dataset(rgb_files, info_dir, img_dir, csv_dir, patch_size, city_name):
    df = pd.DataFrame(columns=['filenames', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax', 'train_test'])

    for rgb_file_name in rgb_files:
        file_name = os.path.basename(rgb_file_name[:-4])
        city_id = int(file_name.split('_')[-1])
        if city_id <= 3:
            print('Processing file {} in validation set'.format(file_name))
            train_test = 'test'
        else:
            print('Processing file {} in training set'.format(file_name))
            train_test = 'train'
        rgb = ersa_utils.load_file(rgb_file_name)
        npy_file_name = os.path.join(info_dir, os.path.basename(rgb_file_name[:-4] + '.npy'))
        coords = ersa_utils.load_file(npy_file_name)

        patch_cnt = 0
        record_cnt = 0
        for line in coords:
            for cell in line:
                patch_cnt += 1
                img_name = os.path.basename(rgb_file_name[:-4] + '_{}.png'.format(patch_cnt))
                save_name = os.path.join(img_dir, img_name)
                img = rgb[cell['h']:cell['h'] + patch_size[0], cell['w']:cell['w'] + patch_size[1], :]
                label = cell['label']
                box = cell['box']
                ersa_utils.save_file(save_name, img)

                if len(box) > 0:
                    for lbl, bbox in zip(label, box):
                        df.loc[patch_cnt] = [img_name, patch_size[0], patch_size[1], lbl,
                                             bbox[1], bbox[0], bbox[3], bbox[2], train_test]
                        record_cnt += 1
    df.to_csv(os.path.join(csv_dir, 'labels_{}.csv'.format(city_name)), index=False)


if __name__ == '__main__':
    # settings
    is_dcc= False
    patch_size = (500, 500)
    encoder = {'DT': 1, 'TT': 2}

    for city_name in ['Tucson', 'Colwich', 'Clyde', 'Wilmington']:
        make_dataset(is_dcc, city_name, patch_size, encoder)
