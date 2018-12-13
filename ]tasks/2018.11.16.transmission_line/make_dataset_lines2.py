import os
import cv2
import numpy as np
import pandas as pd
from glob import glob
from natsort import natsorted
from skimage import measure
from skimage.draw import line, polygon
from PIL import Image, ImageDraw
import ersa_utils
from reader import reader_utils
from preprocess import patchExtractor


# Settings
ENCODER = {'DL': 1, 'TL': 1, 'DT': 2, 'TT': 2}


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
        if group['Type'].values[0] == 'Line' and label in ENCODER:
            for j in range(group.shape[0] - 1):
                r0, c0 = int(group['X'].values[j]), int(group['Y'].values[j])
                r1, c1 = int(group['X'].values[j + 1]), int(group['Y'].values[j + 1])
                cc, rr = line(r0, c0, r1, c1)
                cc, rr = check_bounds(cc, rr, df['width'][0] - 1, df['height'][0] - 1)
                yield label, cc, rr


def read_polygon_csv_data_towers(csv_file):
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


def get_lines(img, patch_size):
    img_binary = (img > 0).astype(np.uint8)
    lbl = measure.label(img_binary)
    props = measure.regionprops(lbl)
    vert_list = []
    # get vertices
    for rp in props:
        vert_list.append(rp.centroid)

    # add lines
    im = Image.new('L', patch_size.tolist())
    for i in range(len(vert_list)):
        for j in range(i+1, len(vert_list)):
            ImageDraw.Draw(im).line((vert_list[i][1], vert_list[i][0],
                                     vert_list[j][1], vert_list[j][0]), fill=1, width=15)
    im_lines = (np.array(im, dtype=float) - img.astype(float) > 0).astype(np.uint8)
    im = (img + im_lines).astype(np.uint8)
    assert set(np.unique(im).tolist()).issubset({0, 1, 2})
    return im


def write_data_info(rgb_files, csv_files, save_dir, patch_size, overlap, pad):
    f_name_list = []
    for rgb_file, csv_file in zip(rgb_files, csv_files):
        print('Processing data {} ...'.format(os.path.basename(rgb_file)[:-3]), end='')
        city_name = os.path.basename(rgb_file[:-4]).split('_')[2]
        city_id = os.path.basename(rgb_file[:-4]).split('_')[-1]

        rgb = ersa_utils.load_file(rgb_file)
        h, w = rgb.shape[:2]
        gt = np.zeros((h, w), dtype=np.uint8)
        gt_towers = np.zeros((h, w), dtype=np.uint8)

        for label, y, x in read_polygon_csv_data(csv_file):
            gt[x, y] = ENCODER[label]

        for label, y, x in read_polygon_csv_data_towers(csv_file):
            gt_towers[y, x] = ENCODER[label]

        # dilation
        kernel = np.ones((15, 15), np.uint8)
        gt = cv2.dilate(gt, kernel, iterations=1)
        tile_size = gt.shape[:2]

        grid = patchExtractor.make_grid(tile_size, patch_size, overlap)
        block = np.dstack([rgb, gt, gt_towers])

        for patch, y, x in patchExtractor.patch_block(block, pad, grid, patch_size, return_coord=True):
            patch = reader_utils.resize_image(patch, patch_size//2, preserve_range=True).astype(np.uint8)
            r_patch = patch[:, :, 0]
            g_patch = patch[:, :, 1]
            b_patch = patch[:, :, 2]
            gt_patch = patch[:, :, 3]
            tw_patch = patch[:, :, 4]
            tw_patch = get_lines(tw_patch, patch_size // 2)

            # save files
            prefix = '{}{}_y{}x{}_'.format(city_name, city_id, y, x)
            ersa_utils.save_file(os.path.join(save_dir, '{}_RGB0.jpg'.format(prefix)), r_patch)
            ersa_utils.save_file(os.path.join(save_dir, '{}_RGB1.jpg'.format(prefix)), g_patch)
            ersa_utils.save_file(os.path.join(save_dir, '{}_RGB2.jpg'.format(prefix)), b_patch)
            ersa_utils.save_file(os.path.join(save_dir, '{}_TW.jpg'.format(prefix)), tw_patch)
            ersa_utils.save_file(os.path.join(save_dir, '{}_GT.png'.format(prefix)), gt_patch)

            f_line = '{}_RGB0.jpg {}_RGB1.jpg {}_RGB2.jpg {}_TW.jpg {}_GT.png\n'.\
                format(prefix, prefix, prefix, prefix, prefix)
            f_name_list.append(f_line)
        print('Done!')

    file_name = os.path.join(save_dir, 'fileList.txt')
    ersa_utils.save_file(file_name, f_name_list)


data_dir = r'/home/lab/Documents/bohao/data/transmission_line/raw'
# get files
rgb_files = natsorted([a for a in glob(os.path.join(data_dir, '*.tif'))
                       if 'multiclass' not in a])
csv_files = natsorted(glob(os.path.join(data_dir, '*.csv')))
save_dir = r'/home/lab/Documents/bohao/data/lines_patches'
ersa_utils.make_dir_if_not_exist(save_dir)
patch_size = np.array([572*2, 572*2])
overlap = 300
pad = 92

write_data_info(rgb_files, csv_files, save_dir, patch_size, overlap, pad)
