import os
import imageio
import numpy as np
from glob import glob
from util_functions import add_mask
import sis_utils

deeplab_pred_dir = r'/hdd/Results/gbdx_cmp/DeeplabV3_res101_inria_aug_grid_0_PS(321, 321)_BS5_EP100_LR1e-05_DS40_DR0.1_SFN32/sp/pred'
unet_pred_dir = r'/hdd/Results/gbdx_cmp/UnetCrop_inria_aug_grid_0_PS(572, 572)_BS5_EP100_LR0.0001_DS60_DR0.1_SFN32/sp/pred'
gt_dir = r'/media/ei-edl01/data/uab_datasets/sp/DATA_BUILDING_AND_PANEL'
tile_ids = [os.path.basename(a).split('_')[0] for a in glob(os.path.join(gt_dir, '*_GT.png'))]
img_dir, task_dir = sis_utils.get_task_img_folder()

for img_cnt in range(len(tile_ids)):
    img_name = '{}.png'.format(tile_ids[img_cnt])
    rgb_name = '{}_RGB.jpg'.format(tile_ids[img_cnt])
    unet_pred = imageio.imread(os.path.join(unet_pred_dir, img_name))
    deeplab_pred = imageio.imread(os.path.join(deeplab_pred_dir, img_name))
    ensemble_pred = unet_pred + deeplab_pred - unet_pred * deeplab_pred
    sp_rgb = imageio.imread(os.path.join(gt_dir, rgb_name))

    masked = add_mask(sp_rgb, ensemble_pred, [255, None, None], mask_1=1)
    img_name = os.path.join(img_dir, 'ca_building', 'ensemble', '{}_building_mask.png'.format(tile_ids[img_cnt]))
    imageio.imsave(img_name, masked)
