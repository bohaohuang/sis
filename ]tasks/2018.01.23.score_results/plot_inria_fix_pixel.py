import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import utils

model_name = ['deeplab', 'unet']
ds_name = ['inria', 'spca']
save_dir = r'/media/ei-edl01/user/bh163/tasks/2018.01.23.score_results'
cmap = plt.get_cmap('Set1').colors

plt.rcParams.update({'font.size': 18})
fig = plt.figure(figsize=(12, 5))
ax1 = plt.subplot(211)
for cnt_1, m in enumerate(model_name):
    file_name = os.path.join(save_dir, '{}_{}_fixpixel.npy'.format(m, 'inria'))
    with open(file_name, 'rb') as pk:
        [result_mean, result_var, result_up, result_down, batch_sizes, patch_sizes] \
            = pickle.load(pk)
    ax1.errorbar(patch_sizes, result_mean, color=cmap[cnt_1], yerr=result_var, uplims=result_up, lolims=result_down,
                 label='{}'.format(m))
plt.grid('on')
for tic in ax1.xaxis.get_major_ticks():
    tic.tick1On = tic.tick2On = False
    tic.label1On = tic.label2On = False

plt.ylabel('Mean IoU')
plt.title('Fix Pixel on Inria')
plt.legend()

ax2 = plt.subplot(212, sharex=ax1)
for cnt_1, m in enumerate(model_name):
    file_name = os.path.join(save_dir, '{}_{}_fixpixel.npy'.format(m, 'spca'))
    with open(file_name, 'rb') as pk:
        [result_mean, result_var, result_up, result_down, batch_sizes, patch_sizes] \
            = pickle.load(pk)
    ind = np.arange(len(batch_sizes))
    ax2.errorbar(patch_sizes, result_mean, color=cmap[cnt_1], yerr=result_var, uplims=result_up, lolims=result_down,
                 label='{}'.format(m))
plt.grid('on')
plt.ylabel('Mean IoU')
plt.title('Fix Pixel on Sp')

plt.xlabel('Patch Size')
plt.tight_layout()
img_dir, task_dir = utils.get_task_img_folder()
plt.savefig(os.path.join(img_dir, 'inria_fixpixel_all.png'))
plt.show()
