import os
import imageio
from glob import glob
import matplotlib.pyplot as plt

img_dir = r'/hdd/uab_datasets/Results/PatchExtr/inria/chipExtrReg_cSz572x572_pad184'
#files = sorted(glob(os.path.join(img_dir, '*GT_Divide.png')))

x = 1440
y = 1920
size = 572
rgb_dir = r'/media/ei-edl01/data/uab_datasets/inria/data/Original_Tiles'
rgb = imageio.imread(os.path.join(rgb_dir, 'austin1_RGB.tif'))
gt = imageio.imread(os.path.join(rgb_dir, 'austin1_GT.tif'))
pred = imageio.imread(r'/hdd/Results/UnetCrop_inria_aug_grid_1_PS(572, 572)_BS5_EP100_LR0.0001_DS60_DR0.1_SFN32/default/pred/austin1.png')
gt_patch = gt[x:x+size, y:y+size]
pred_patch = pred[x:x+size, y:y+size]
rgb_patch = rgb[x:x+size, y:y+size, :]

save_dir = r'/media/ei-edl01/user/bh163/figs/2017.12.27.igarss_figures/jordan'

imageio.imsave(os.path.join(save_dir, 'rgb.png'), rgb)
imageio.imsave(os.path.join(save_dir, 'pred.png'), pred)
imageio.imsave(os.path.join(save_dir, 'rgb_patch.png'), rgb_patch)
imageio.imsave(os.path.join(save_dir, 'pred_patch.png'), pred_patch)
