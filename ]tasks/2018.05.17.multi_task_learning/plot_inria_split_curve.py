import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import utils
import util_functions


img_dir, task_dir = utils.get_task_img_folder()

# plot iou and lr training curves
fields = ['Step', 'Value']
lsty = ['-', '--']
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
plt_label = ['32', '64']
plt.figure(figsize=(8, 4))
for cnt, iou_file_name in \
        enumerate([r'run_UnetCrop_inria_decay_0_PS(572, 572)_BS5_EP100_LR0.0001_DS60.0_DR0.1_SFN32-tag-iou_validation.csv',
                   r'run_.-tag-iou_validation.csv']):
        df = pd.read_csv(os.path.join(task_dir, iou_file_name), skipinitialspace=True, usecols=fields)
        value = util_functions.savitzky_golay(np.array(df['Value']), 7, 2)
        plt.plot(value * 100, label=plt_label[cnt])
plt.grid('on')
plt.ylim([50, 80])
plt.ylabel('IoU')
plt.legend()
plt.xlabel('Epoch')
plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'split_train_curve_cmp.png'))
plt.show()
