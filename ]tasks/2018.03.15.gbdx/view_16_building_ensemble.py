import os
import imageio
import numpy as np
from glob import glob
from util_functions import add_mask
import matplotlib.pyplot as plt

pred_base_dir = r'/media/ei-edl01/user/bh163/figs/2018.03.15.gbdx/ca_building'
gt_dir = r'/media/ei-edl01/data/uab_datasets/sp/DATA_BUILDING_AND_PANEL'
tile_ids = [os.path.basename(a).split('_')[0] for a in glob(os.path.join(pred_base_dir, 'deeplab', '*.png'))]

for img_cnt in range(len(tile_ids)):
    img_name = '{}_building_mask.png'.format(tile_ids[img_cnt])
    gt_name = '{}_GT.png'.format(tile_ids[img_cnt])
    unet_pred = imageio.imread(os.path.join(pred_base_dir, 'unet', img_name))
    deeplab_pred = imageio.imread(os.path.join(pred_base_dir, 'deeplab', img_name))
    ensemble_pred = imageio.imread(os.path.join(pred_base_dir, 'ensemble', img_name))
    sp_gt = imageio.imread(os.path.join(gt_dir, gt_name))

    masked_unet = add_mask(unet_pred, sp_gt, [0, 0, 255], mask_1=1)
    masked_deeplab = add_mask(deeplab_pred, sp_gt, [0, 0, 255], mask_1=1)
    masked_ensemble = add_mask(ensemble_pred, sp_gt, [0, 0, 255], mask_1=1)


    '''plt.figure(figsize=(15, 8))
    ax1 = plt.subplot2grid((2, 3), (0, 0), colspan=2, rowspan=2)
    plt.imshow(masked_ensemble)
    #plt.axis('off')
    plt.title('ENSEMBLE')
    ax2 = plt.subplot2grid((2, 3), (0, 2), sharex=ax1, sharey=ax1)
    plt.imshow(masked_unet)
    #plt.axis('off')
    plt.title('UNET')
    ax3 = plt.subplot2grid((2, 3), (1, 2), sharex=ax1, sharey=ax1)
    plt.imshow(masked_deeplab)
    # plt.axis('off')
    plt.title('DEEPLAB')
    plt.tight_layout()
    plt.show()'''

    plt.figure(figsize=(8, 7))
    plt.imshow(masked_ensemble)
    plt.title(tile_ids[img_cnt])
    plt.tight_layout()
    plt.show()
