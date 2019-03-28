"""

"""


# Built-in
import os
from glob import glob

# Libs
from natsort import natsorted

# Own modules
from quality_assurance import read_lines_truth
from make_dataset_demo_figure import read_tower_truth


def read_data(data_dir, city_name):
    gt_files = get_gt_files(data_dir, city_name)
    tile_num = len(gt_files)
    n_tower = 0
    n_line = 0
    print('City: {}, #Images: {}'.format(city_name, tile_num))
    for gt in gt_files:
        tower_gt = read_tower_truth(gt)
        line_gt = read_lines_truth(gt, tower_gt)
        n_tower += len(tower_gt)
        n_line += len(line_gt)
    print('\t #Tower:{}, #Line:{}'.format(n_tower, n_line))


def get_gt_files(data_dir, city_name):
    return natsorted(glob(os.path.join(data_dir, 'USA_*_{}_*.csv'.format(city_name))))


if __name__ == '__main__':
    city_list = ['Tucson', 'Colwich', 'Clyde', 'Wilmington']
    data_dir = r'/home/lab/Documents/bohao/data/transmission_line/raw'
    for i in range(4):
        read_data(data_dir, city_list[i])
