import os
import cv2
import imageio
import matplotlib.pyplot as plt
import utils
import util_functions

orig_dir = r'/media/ei-edl01/data/uab_datasets/gbdx/data/Original_Tiles'
tile_id = '000795-se'
img_name = os.path.join(orig_dir, '{}_RGB.tif'.format(tile_id))
gt_name = os.path.join(orig_dir, '{}_GT.png'.format(tile_id))
img_dir, task_dir = utils.get_task_img_folder()

img = imageio.imread(img_name)
imageio.imsave(os.path.join(img_dir, 'website_demo_figs', '{}_mask.png'.format(tile_id)), img)
gt = imageio.imread(gt_name)
img_mask = util_functions.add_mask(img, gt, [255, None, None], mask_1=255)
imageio.imsave(os.path.join(img_dir, 'website_demo_figs', '{}_orig.png'.format(tile_id)), img_mask)
