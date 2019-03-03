import os
import numpy as np
import matplotlib.pyplot as plt
import sis_utils
import util_functions

img_dir, task_dir = sis_utils.get_task_img_folder()
base_dir = r'/hdd/Results/inria_gamma/UnetCrop_inria_aug_grid_0_PS(572, 572)_BS5_EP100_LR0.0001_DS60_DR0.1_SFN32'
ft_dir1 = r'/hdd/Results/inria_gamma/UnetCrop_inria_aug_gamma_0_PS(572, 572)_BS5_EP40_LR1e-05_DS20_DR0.1_SFN32'
ft_dir2 = r'/hdd/Results/inria_gamma/UnetCrop_inria_aug_gamma_0_PS(572, 572)_BS5_EP40_LR1e-06_DS20_DR0.1_SFN32'
ts_dir = r'/hdd/Results/inria_gamma/UnetCrop_inria_aug_gamma_0_PS(572, 572)_BS5_EP100_LR0.0001_DS40_DR0.1_SFN32'

model_name = ['base', 'finetune-5', 'finetune-6', 'scratch']
model_dir = [base_dir, ft_dir1, ft_dir2, ts_dir]
gamma_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.5, 2]

plt.rcParams.update({'font.size': 12})
plt.rc('grid', linestyle='--')
for name, directory in zip(model_name, model_dir):
    iou_record = np.zeros(len(gamma_list))
    for cnt, gamma in enumerate(gamma_list):
        result_file = os.path.join(directory, 'ct{}'.format(util_functions.d2s(gamma)), 'result.txt')
        with open(result_file, 'r') as f:
            results = f.readlines()
            iou_record[cnt] = float(results[-1])
    plt.plot(gamma_list, iou_record, '-o', label=name)
plt.title('IoU Vs Gamma')
plt.grid()
plt.xlabel('Gamma')
plt.ylabel('IoU')
plt.legend()
plt.savefig(os.path.join(img_dir, 'iou_vs_gamma_ft_ts.png'))
plt.show()
