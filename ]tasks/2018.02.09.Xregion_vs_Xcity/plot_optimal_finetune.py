import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import utils
import util_functions


plt.rcParams.update({'font.size': 14})

img_dir, task_dir = utils.get_task_img_folder()

lty = ['-', '--', '-.']
cmap = plt.get_cmap('Set1').colors

plt.figure(figsize=(12, 6))
for cnt_1, layer_keep in enumerate(range(5)):
    for cnt_2, learn_rate in enumerate(['1e-05', '1e-06', '1e-07']):
        file_name = r'run_UnetCrop_inria_aug_leave_austin_keep_{}_0_PS(572, 572)_BS5_EP20_LR{}_DS10_DR0.1_SFN32-tag-' \
                    r'iou_validation.csv'.format(layer_keep, learn_rate)
        file = os.path.join(task_dir, file_name)
        df = pd.read_csv(file, skipinitialspace=True, usecols=['Step', 'Value'])
        value = util_functions.savitzky_golay(np.array(df['Value']), 13, 2)
        plt.plot(np.arange(20), value, color=cmap[cnt_1], linestyle=lty[cnt_2],
                 label='layer:{},lr:{}'.format(layer_keep, learn_rate))
plt.legend(ncol=5, fontsize=12)
plt.xlabel('Epoch')
plt.ylabel('IoU')
plt.title('Validation IoU')
plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'finetune_inria.png'))
plt.show()