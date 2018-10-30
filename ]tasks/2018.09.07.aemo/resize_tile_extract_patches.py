import os
import numpy as np
from glob import glob
import ersa_utils
from reader import reader_utils
from preprocess import patchExtractor


def extract_patches(img_pair, patch_size, pad, overlap, patch_dir, prefix, file_exts=('jpg','png'),
                    file_suffix=('rgb', 'gt')):
    grid_list = patchExtractor.make_grid(tile_size + 2 * pad, patch_size, overlap)
    record_file = open(os.path.join(patch_dir, 'file_list.txt'), 'a+')
    patch_list = []
    for suffix_cnt, (img, ext) in enumerate(zip(img_pair, file_exts)):
        patch_list_ext = []
        # extract images
        for patch, y, x in patchExtractor.patch_block(img, pad, grid_list, patch_size, return_coord=True):
            patch_name = '{}_y{}x{}.{}'.format(prefix+'_{}'.format(file_suffix[suffix_cnt]), int(y), int(x), ext)
            patch_name = os.path.join(patch_dir, patch_name)
            ersa_utils.save_file(patch_name, patch)
            patch_list_ext.append(patch_name)
        patch_list.append(patch_list_ext)
    patch_list = ersa_utils.rotate_list(patch_list)
    for items in patch_list:
        record_file.write('{}\n'.format(' '.join(items)))
    record_file.close()


data_dir = r'/home/lab/Documents/bohao/data/aemo/aemo_align'
rgb_files = sorted(glob(os.path.join(data_dir, '*rgb.tif')))
gt_files = sorted(glob(os.path.join(data_dir, '*d255.tif')))
patch_dir = r'/hdd/ersa/patch_extractor/aemo_resize'
patch_size = np.array([572, 572])


gsd_list = [0.372, 0.38, 0.35]
refer_size = 0.3
resize_list = []
for gsd in gsd_list:
    resize_list.append(int(5000 * gsd / refer_size))

for cnt, (rgb_file, gt_file) in enumerate(zip(rgb_files, gt_files)):
    tile_size = np.array([resize_list[cnt//2], resize_list[cnt//2]])

    rgb = ersa_utils.load_file(rgb_file)
    gt = ersa_utils.load_file(gt_file)
    rgb = reader_utils.resize_image(rgb, tile_size)
    gt = reader_utils.resize_image(gt, tile_size)

    prefix = os.path.basename(rgb_file).split('.')[0][:-4]
    print('Extracting patches for {}...'.format(prefix))
    extract_patches((rgb, gt), patch_size, 92, 184, patch_dir, prefix)
    print('Done!')
