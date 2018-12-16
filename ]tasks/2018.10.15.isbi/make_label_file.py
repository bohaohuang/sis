import os
import numpy as np
import pandas as pd
from glob import glob
from natsort import natsorted
from skimage.draw import polygon
import ersa_utils


def check_bounds(r, w, size_y, size_x):
    """
    Check if the mask exceeds image bounds
    :param r: rows of the mask
    :param w: cols of the mask
    :param size_y: height of the patch
    :param size_x: width of the patch
    :return: refined mask after bounds checking
    """
    r = np.maximum(np.minimum(r, size_y), 0)
    w = np.maximum(np.minimum(w, size_x), 0)
    return r, w


def create_polygon_fig(csv_files, patch_size=(300, 300)):
    """
    Create and save the polygon ground truth image
    :param csv_files: list of csv files
    :param patch_size: used if the ground truth file is empty
    :return:
    """
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)

        if not df.empty:
            gt = np.zeros((df['height'][0], df['width'][0]))
            for name, group in df.groupby('Object', sort=False):
                y, x = polygon(group['Y'].values, group['X'].values)
                y, x = check_bounds(y, x, df['height'][0] - 1, df['width'][0] - 1)
                gt[y, x] = 1
        else:
            gt = np.zeros((patch_size))

        save_name = csv_file[:-3] + 'png'
        ersa_utils.save_file(save_name, gt.astype(np.uint8))


if __name__ == '__main__':
    data_dir = r'/media/ei-edl01/data/uab_datasets/bihar/labeled patches'
    csv_files = natsorted(glob(os.path.join(data_dir, '*.csv')))

    create_polygon_fig(csv_files)
