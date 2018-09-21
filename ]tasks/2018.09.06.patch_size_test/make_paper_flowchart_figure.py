import os
import numpy as np
import ersa_utils

unet_pred_dir = r'/hdd/Results/domain_selection/UnetCrop_inria_aug_grid_0_PS(572, 572)_BS5_EP100_LR0.0001_DS60_DR0.1_SFN32/inria/pred'
ugan_pred_dir = r'/hdd/Results/ugan/UnetGAN_V3_inria_gan_xregion_0_PS(572, 572)_BS20_EP30_LR0.0001_1e-06_1e-06_DS30.0_30.0_30.0_DR0.1_0.1_0.1/inria/pred'
save_dir = r'/media/ei-edl01/user/bh163/temp/figs'

file_name = 'chicago2.png'

unet_pred_file = os.path.join(unet_pred_dir, file_name)
ugan_pred_file = os.path.join(ugan_pred_dir, file_name)

unet_pred = ersa_utils.load_file(unet_pred_file)
ugan_pred = ersa_utils.load_file(ugan_pred_file)

ersa_utils.save_file(os.path.join(save_dir, 'unet_{}'.format(file_name)), (unet_pred).astype(np.uint8))
ersa_utils.save_file(os.path.join(save_dir, 'ugan_{}'.format(file_name)), (ugan_pred*255).astype(np.uint8))
