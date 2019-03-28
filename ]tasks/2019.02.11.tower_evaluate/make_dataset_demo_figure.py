"""

"""


# Built-in
import os

# Libs
import cv2
import imageio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.draw import polygon

# Own modules


def add_mask(orig_img, mask, mask_val, mask_1=1):
    mask_locs = np.where(mask == mask_1)
    locs_num = len(mask_locs[0])
    if mask_val[0] is not None:
        orig_img[mask_locs[0], mask_locs[1], np.zeros(locs_num, dtype=int)] = mask_val[0]
    if mask_val[1] is not None:
        orig_img[mask_locs[0], mask_locs[1], np.ones(locs_num, dtype=int)] = mask_val[1]
    if mask_val[2] is not None:
        orig_img[mask_locs[0], mask_locs[1], 2*np.ones(locs_num, dtype=int)] = mask_val[2]
    return orig_img


def read_polygon_csv_data(csv_file):
    """
    Read polygons from annotation files
    :param csv_file: the path to the annotation file
    :return: minimum bounding boxes for each polygon
    """
    def get_bounding_box(y, x):
        y_min = np.min(y).astype(int)
        x_min = np.min(x).astype(int)
        y_max = np.max(y).astype(int)
        x_max = np.max(x).astype(int)
        return y_min, x_min, y_max, x_max

    encoder = {'DT': 1, 'TT': 2}
    label_order = ['SS', 'OT', 'DT', 'TT', 'OL', 'DL', 'TL']
    df = pd.read_csv(csv_file)
    df['temp_label'] = pd.Categorical(df['Label'], categories=label_order, ordered=True)
    df.sort_values('temp_label', inplace=True, kind='mergesort')

    for name, group in df.groupby('Object', sort=False):
        label = group['Label'].values[0]
        if group['Type'].values[0] == 'Polygon' and label in encoder:
            x, y = polygon(group['X'].values, group['Y'].values)
            yield label, get_bounding_box(y, x)


def get_center_point(ymin, xmin, ymax, xmax):
    """
    Get the center point of a bounding box, we are only scoring the centers not the bbox
    :param ymin:
    :param xmin:
    :param ymax:
    :param xmax:
    :return: the center point coordinates
    """
    return ((ymin+ymax)/2, (xmin+xmax)/2)


def read_tower_truth(csv_file_name):
    """
    Get the coordinates information of the towers
    :param csv_file_name: the path to the annotation file
    :return: a list of coordinates
    """
    gt_list = []
    for label, bbox in read_polygon_csv_data(csv_file_name):
        y, x = get_center_point(*bbox)
        gt_list.append([y, x])
    return gt_list


def read_data(raw_dir, line_dir, city_name, tile_id):
    """
    Read raw image, tower truth and line truth
    :param raw_dir:
    :param line_dir:
    :param city_name:
    :param tile_id:
    :return:
    """
    rgb_file = os.path.join(raw_dir, 'USA_{}_{}.tif'.format(city_name, tile_id))
    tower_file = os.path.join(raw_dir, 'USA_{}_{}.csv'.format(city_name, tile_id))
    line_file = os.path.join(line_dir, '{}{}_GT.png'.format(city_name.split('_')[1], tile_id))
    img = imageio.imread(rgb_file)
    tower_gt = read_tower_truth(tower_file)
    line_gt = imageio.imread(line_file)
    return img, tower_gt, line_gt


def visualize_image(img, tower_gt, line_gt, line_color, tower_color, size, marker, alpha, edgecolor):
    plt.figure(figsize=(8.5, 8))
    line_gt = cv2.dilate(line_gt, np.ones((5, 5), np.uint8), iterations=10)
    img_with_line = add_mask(img, line_gt, line_color, 1)
    plt.imshow(img_with_line)
    center_points = np.array(tower_gt)
    plt.scatter(center_points[:, 1], center_points[:, 0], c=tower_color, s=size, marker=marker, alpha=alpha,
                edgecolors=edgecolor)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    city_list = ['AZ_Tucson', 'KS_Colwich_Maize', 'NC_Clyde', 'NC_Wilmington']
    city_id = 0     # which city in the city_list to plot
    tile_id = 1     # the tile number, should be from 1 to N (N=maximum number in the city)

    raw_dir = r'/home/lab/Documents/bohao/data/transmission_line/raw'
    line_dir = r'/media/ei-edl01/data/uab_datasets/lines/data/Original_Tiles'

    img, tower_gt, line_gt = read_data(raw_dir, line_dir, city_list[city_id], tile_id)
    visualize_image(img, tower_gt, line_gt,
                    line_color=[0, 255, 0],     # rgb color of the lines in the plot
                    tower_color='r',            # marker color of the towers
                    size=40,                    # marker size of the towers
                    marker='o',                 # marker style of the towers
                    alpha=1,                    # alpha of ther tower markers
                    edgecolor='k')              # edge color of the tower markers
