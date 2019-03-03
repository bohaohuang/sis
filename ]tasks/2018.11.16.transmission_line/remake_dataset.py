import os
import numpy as np
from glob import glob
import sis_utils
import ersa_utils

data_dir = r'/home/lab/Documents/bohao/data/transmission_line'
city_list = ['Tucson', 'Colwich', 'Clyde', 'Wilmington']
save_dir = r'/media/ei-edl01/data/uab_datasets/infrastructure/data/Original_Tiles'
img_dir, task_dir = sis_utils.get_task_img_folder()

'''for city_id in range(4):
    gt_files = sorted(glob(os.path.join(data_dir, '*{}*_multiclass.tif'.format(city_list[city_id]))))
    rgb_files = ['_'.join(a.split('_')[:-1])+'.tif' for a in gt_files]
    for rgb_file_name, gt_file_name in zip(rgb_files, gt_files):
        rgb = ersa_utils.load_file(rgb_file_name)
        gt = ersa_utils.load_file(gt_file_name)

        city_name = os.path.basename(rgb_file_name)[7:].split('_')[0]
        tile_id = int(os.path.basename(rgb_file_name).split('.')[0].split('_')[-1])

        gt[np.where(gt == 7)] = 5
        ersa_utils.save_file(os.path.join(save_dir, '{}{}_RGB.tif'.format(city_name, tile_id)), rgb.astype(np.uint8))
        ersa_utils.save_file(os.path.join(save_dir, '{}{}_GT.png'.format(city_name, tile_id)), gt.astype(np.uint8))
        print('Saved {}_{}'.format(city_name, tile_id))'''

# check gt's
'''gt_files = glob(os.path.join(save_dir, '*.png'))
for g in gt_files:
    gt = ersa_utils.load_file(g)
    print(np.unique(gt))'''
