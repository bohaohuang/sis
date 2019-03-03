import os
import matplotlib.pyplot as plt
import sis_utils
import ersa_utils

img_dir, task_dir = sis_utils.get_task_img_folder()

# plot train all curves
data_type = ['Orig', 'Hist']
lrs = [0.001, 0.0001]
run_ids = range(4)
marker_style = ['o', 'd']
line_style = ['-', '--']
colors = ersa_utils.get_default_colors()

plt.figure(figsize=(12, 6))
for d_cnt, dt in enumerate(data_type):
    for l_cnt, lr in enumerate(lrs):
        for run_id in run_ids:
            if dt == 'Orig':
                file_name = 'run_unet_aemo_{}_PS(572, 572)_BS5_EP80_' \
                            'LR{}_DS30_DR0.1-tag-IoU.csv'.format(run_id, lr)
            else:
                file_name = 'run_unet_aemo_hist_{}_hist_PS(572, 572)_BS5_EP80_' \
                            'LR{}_DS30_DR0.1-tag-IoU.csv'.format(run_id, lr)
            step, value = ersa_utils.read_tensorboard_csv(os.path.join(task_dir, file_name))
            plt.plot(step, value, marker=marker_style[d_cnt], linestyle=line_style[l_cnt], color=colors[d_cnt*2+l_cnt],
                     label='{}{}_{}'.format(dt, lr, run_id))
plt.legend(ncol=4)
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'train_all_curve_cmp.png'))
plt.show()

# plot train later all curves
train_type = ['F7', 'F8']
lrs = [0.001, 0.0001]
run_ids = range(4)
plt.figure(figsize=(12, 6))
for d_cnt, dt in enumerate(train_type):
    for l_cnt, lr in enumerate(lrs):
        for run_id in run_ids:
            file_name = 'run_unet_aemo_hist_{}_{}_hist_PS(572, 572)_BS5_EP80_LR{}_' \
                        'DS30_DR0.1-tag-IoU.csv'.format('up{}'.format(7 + d_cnt), run_id, lr)

            step, value = ersa_utils.read_tensorboard_csv(os.path.join(task_dir, file_name))
            plt.plot(step, value, marker=marker_style[d_cnt], linestyle=line_style[l_cnt],
                     color=colors[d_cnt * 2 + l_cnt],
                     label='{}{}_{}'.format(dt, lr, run_id))
plt.legend(ncol=4)
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'train_later_curve_cmp.png'))
plt.show()
