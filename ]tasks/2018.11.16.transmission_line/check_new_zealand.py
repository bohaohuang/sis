"""

"""


# Built-in
import os
from glob import glob

# Libs
from natsort import natsorted

# Own modules
from rst_utils import visualize
from rst_utils import misc_utils


def main():
    data_dir = r'/media/ei-edl01/data/remote_sensing_data/transmission_line/New_Zealand'
    city_list = ['Gisborne', 'Palmertson', 'Rotorua', 'Tauranga']
    for city in city_list:
        image_dir = os.path.join(data_dir, 'New_Zealand_{}'.format(city))
        img_list = glob(os.path.join(image_dir, '*.tif'))
        rgb_list = natsorted([a for a in img_list if 'multiclass' not in a and 'resize' not in a])
        gt_list = natsorted([a for a in img_list if 'multiclass' in a])
        assert len(rgb_list) == len(gt_list)
        for rgb_file, gt_file in zip(rgb_list, gt_list):
            print(os.path.basename(rgb_file), os.path.basename(gt_file))
            rgb = misc_utils.load_file(rgb_file)
            gt = misc_utils.load_file(gt_file)
            visualize.compare_figures([rgb, gt], (1, 2), fig_size=(12, 5))


if __name__ == '__main__':
    main()
