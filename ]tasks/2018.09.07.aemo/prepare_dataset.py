import os
from glob import glob
from shutil import copyfile
from tqdm import tqdm

orig_dir = r'/media/ei-edl01/data/aemo/TILES_WITH_GT'
set_ids = ['10', '30', '50']
dest_dir = r'/home/lab/Documents/bohao/data/aemo_temp'

for set_id in tqdm(set_ids):
    data_dir = os.path.join(orig_dir, r'0584270470{}_01'.format(set_id))
    gt_files = sorted(glob(os.path.join(data_dir, '*_gt.tif')))

    rgb_files = sorted(glob(os.path.join(data_dir, '*.tif')))
    rgb_files = [f for f in rgb_files if f not in gt_files]

    for rgb, gt in zip(rgb_files, gt_files):
        rgb_name = os.path.basename(rgb)
        gt_name = os.path.basename(gt)

        new_rgb_name = rgb_name.replace('aus', 'aus{}'.format(set_id))
        new_rgb_name = str(new_rgb_name.split('.')[0]) + '_rgb.tif'
        new_gt_name = gt_name.replace('aus', 'aus{}'.format(set_id))

        copyfile(rgb, os.path.join(dest_dir, new_rgb_name))
        copyfile(gt, os.path.join(dest_dir, new_gt_name))
