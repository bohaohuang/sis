"""

"""


# Built-in
import os
from glob import glob

# Libs
from natsort import natsorted

# Own modules
import ersa_utils
from quality_assurance import read_lines_truth
from make_dataset_demo_figure import read_tower_truth


def read_data(data_dir, city_name):
    gt_files = get_gt_files2(data_dir, city_name)
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
    city_area = 0
    for rgb in get_rgb_files(data_dir, city_name):
        img = ersa_utils.load_file(rgb)
        city_area += (img.shape[0] * img.shape[1]) * 0.3 * 1e-3 * 0.3 * 1e-3
    print(city_area)


def get_gt_files(data_dir, city_name):
    return natsorted(glob(os.path.join(data_dir, 'USA_*_{}_*.csv'.format(city_name))))

def get_gt_files2(data_dir, city_name):
    temp_list = natsorted(glob(os.path.join(data_dir, '*.csv')))
    city_list = []
    for f in temp_list:
        if city_name in f:
            if 'NZ' in f:
                if 'resize' in f:
                    city_list.append(f)
            else:
                city_list.append(f)
    return city_list


def get_rgb_files(data_dir, city_name):
    temp_list = natsorted(glob(os.path.join(data_dir, '*class.tif')))
    city_list = []
    for f in temp_list:
        if city_name in f:
            if 'NZ' in f:
                if 'resize' in f:
                    city_list.append(f)
            else:
                city_list.append(f)
    return city_list


if __name__ == '__main__':
    city_list = ['Tucson', 'Colwich', 'Dunedin', 'Gisborne', 'Palmerston', 'Rotorua', 'Tauranga']
    data_dir = r'/home/lab/Documents/bohao/data/transmission_line/raw2'
    for i in range(len(city_list)):
        read_data(data_dir, city_list[i])
