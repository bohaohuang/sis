import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import utils
import util_functions

img_dir, task_dir = utils.get_task_img_folder()
fields = ['Step', 'Value']

plt.rcParams.update({'font.size': 16})
plt.figure(figsize=(8, 4))
files = ['run_UnetCrop_inria_aug_xent_valiou_0_PS(572, 572)_BS5_EP100_LR0.0001_DS60_DR0.1_SFN32-tag-iou_validation.csv',
         'run_UnetCrop_inria_aug_focal_0_g0_a0.5_PS(572, 572)_BS5_EP100_LR0.0001_DS60_DR0.1_SFN32-tag-iou_validation.csv']
label_str = ['Xent', 'Focal']
lsty = ['--', '-']
for cnt, file in enumerate(files):
    df = pd.read_csv(os.path.join(task_dir, file), skipinitialspace=True, usecols=fields)
    value = util_functions.savitzky_golay(np.array(df['Value']), 11, 2)
    plt.plot(np.arange(100), value, linestyle=lsty[cnt], label=label_str[cnt])
plt.xlabel('Epoch')
plt.ylabel('IoU')
plt.legend()
plt.grid('on')
plt.title('Validation IoU in Training')
plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'focal_cmp_inria_baseline.png'))
plt.show()
