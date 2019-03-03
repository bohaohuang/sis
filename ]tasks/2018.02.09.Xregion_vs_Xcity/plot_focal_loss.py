import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc
import sis_utils
import util_functions


rc('text', usetex=True)
plt.rcParams.update({'font.size': 14})

img_dir, task_dir = sis_utils.get_task_img_folder()
names = ['run_UnetCrop_spca_aug_focal_valiou_0_g1.0_a0.0005_PS(572, 572)_BS5_EP100_LR0.0001_DS60_DR0.1_SFN32-tag-iou_validation.csv',
         'run_UnetCrop_spca_aug_focal_valiou_0_g2_a0.0005_PS(572, 572)_BS5_EP100_LR0.0001_DS60_DR0.1_SFN32-tag-iou_validation.csv',
         'run_UnetCrop_spca_aug_focal_valiou_0_g5.0_a0.0005_PS(572, 572)_BS5_EP100_LR0.0001_DS60_DR0.1_SFN32-tag-iou_validation.csv',
         'run_UnetCrop_spca_aug_grid_0_PS(572, 572)_BS5_EP100_LR0.0001_DS60_DR0.1_SFN32-tag-iou_validation.csv']

label_str = [r'$\alpha=5e^{-4};\gamma=1$',
             r'$\alpha=5e^{-4};\gamma=2$',
             r'$\alpha=5e^{-4};\gamma=5$',
             '$XRegion$']
lty = ['-', '-', '-', '--']

for cnt, file in enumerate(names):
    file = os.path.join(task_dir, file)
    df = pd.read_csv(file, skipinitialspace=True, usecols=['Step', 'Value'])
    value = util_functions.savitzky_golay(np.array(df['Value']), 17, 2)
    plt.plot(np.arange(100), value, linestyle=lty[cnt], label=label_str[cnt])
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('IoU')
plt.title('Validation IoU')
plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'focal_spca.png'))
plt.show()
