import os
import imageio
import utils

img_dir = r'/hdd/Results/domain_selection/UnetCrop_inria_aug_leave_0_0_PS(572, 572)_BS5_EP100_LR0.0001_' \
          r'DS60_DR0.1_SFN32/inria/pred/austin1.png'
img = imageio.imread(img_dir)

img_dir, task_dir = utils.get_task_img_folder()
imageio.imsave(os.path.join(img_dir, 'austin1.png'), img*255)
