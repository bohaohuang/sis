import os
import time
import numpy as np
from glob import glob
import ersa_utils


def get_blank_regions(img):
    img_tmp = img.astype(np.float32)
    img_tmp = np.sum(img_tmp, axis=2)
    blank_mask = (img_tmp < 0.1).astype(np.int)
    return blank_mask


tile_name = ['aus10_0x0_gt_d255', 'aus10_4453x10891_gt_d255', 'aus30_4559x10766_gt_d255',
             'aus30_13678x10766_gt_d255', 'aus50_0x3179_gt_d255', 'aus50_4808x3179_gt_d255']
pred_dir = r'/hdd/Results/aemo/uab/UnetCrop_aemo_0_all_PS(572, 572)_BS5_EP80_LR0.001_DS30_DR0.1_SFN32/aemo_comb/pred'
pred_files = sorted(glob(os.path.join(pred_dir, '*.png')))
rgb_dir = r'/home/lab/Documents/bohao/data/aemo'

save_dir = os.path.join(r'/media/ei-edl01/data/aemo/pred', time.strftime('%Y_%m_%d', time.gmtime()))
ersa_utils.make_dir_if_not_exist(save_dir)

for tn, pf in zip(tile_name, pred_files):
    rgb_file = os.path.join(rgb_dir, '{}_rgb.tif'.format(tn[:-8]))
    rgb = ersa_utils.load_file(rgb_file)
    bm = 1 - get_blank_regions(rgb)

    pred = ersa_utils.load_file(pf)
    pred = pred * bm

    save_pred_name = '{}_pred.png'.format(tn[:-8])
    ersa_utils.save_file(os.path.join(save_dir, save_pred_name), pred)
