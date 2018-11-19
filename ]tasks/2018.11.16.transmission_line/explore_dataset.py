import os
import numpy as np
from glob import glob
import utils
import ersa_utils
from visualize import visualize_utils

data_dir = r'/home/lab/Documents/bohao/data/transmission_line'
city_list = ['Tucson', 'Colwich', 'Clyde', 'Wilmington']
city_id = 3
img_dir, task_dir = utils.get_task_img_folder()

gt_files = sorted(glob(os.path.join(data_dir, '*{}*_multiclass.tif'.format(city_list[city_id]))))
rgb_files = ['_'.join(a.split('_')[:-1])+'.tif' for a in gt_files]
for rgb_file_name, gt_file_name in zip(rgb_files, gt_files):
    print(rgb_file_name, gt_file_name)
    rgb = ersa_utils.load_file(rgb_file_name)
    gt = ersa_utils.load_file(gt_file_name)
    print(rgb.shape, gt.shape)

    print(np.unique(gt))
    visualize_utils.compare_two_figure(rgb, gt)
