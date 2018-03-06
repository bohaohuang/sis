import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import utils

model_name = ['deeplab', 'unet', 'frrn']
save_dir = r'/media/ei-edl01/user/bh163/tasks/2018.01.23.score_results'

plt.rcParams.update({'font.size': 18})
fig = plt.figure(figsize=(12, 5))
for m in model_name:
    file_name = os.path.join(save_dir, '{}_inria_fixpixel.npy'.format(m))
    with open(file_name, 'rb') as pk:
        [result_mean, result_var, result_up, result_down, batch_sizes, patch_sizes] \
            = pickle.load(pk)
    ind = np.arange(len(batch_sizes))
    ax1 = plt.subplot()
    ax1.errorbar(patch_sizes, result_mean, yerr=result_var, uplims=result_up, lolims=result_down,
                 label='{}'.format(m))
plt.grid('on')
plt.xlabel('Patch Size')
plt.ylabel('Mean IoU')
plt.title('Fix Pixel on Inria')
plt.legend()
plt.tight_layout()
img_dir, task_dir = utils.get_task_img_folder()
plt.savefig(os.path.join(img_dir, 'inria_fixpixel_all.png'))
plt.show()
