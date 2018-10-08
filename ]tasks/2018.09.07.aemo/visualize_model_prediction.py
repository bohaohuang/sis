import os
import numpy as np
from glob import glob
import ersa_utils
from visualize import visualize_utils

ds = 40
pred_dir = r'/hdd/Results/aemo/unet_aemo_PS(572, 572)_BS5_EP60_LR0.001_DS{}_DR0.1/default/pred'.format(ds)
orig_dir = r'/home/lab/Documents/bohao/data/aemo'

pred_files = sorted(glob(os.path.join(pred_dir, '*.png')))
rgb_files = sorted(glob(os.path.join(orig_dir, '*rgb.tif')))[-2:]
gt_files = sorted(glob(os.path.join(orig_dir, '*gt.tif')))[-2:]

for (pred, rgb, gt) in zip(pred_files, rgb_files, gt_files):
    print(pred, rgb, gt)
    p = ersa_utils.load_file(pred)
    r = ersa_utils.load_file(rgb)
    g = (ersa_utils.load_file(gt) / 255).astype(np.uint8)

    print(np.unique(g), np.unique(p))

    diff = g.astype(np.int32) - p.astype(np.int32)
    print(np.unique(diff))
    visualize_utils.compare_two_figure(r, diff)
