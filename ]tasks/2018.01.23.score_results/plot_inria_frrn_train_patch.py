import os
import utils
import numpy as np


img_dir, task_dir = utils.get_task_img_folder()
save_dir = os.path.join(task_dir, 'train_patch')

iou, duration = np.load(os.path.join(save_dir, r'FRRN_inria_aug_grid_1_PS(224, 224)_BS5_EP100_LR0.0001_DS60_DR0.1_SFN32_224_0.npy'))
print(iou)
print(duration)
