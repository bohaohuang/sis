import os
import numpy as np
import matplotlib.pyplot as plt
import sis_utils
import util_functions

img_dir, task_dir = sis_utils.get_task_img_folder()
deeplab_dir = r'/hdd/Results/ct_gamma/DeeplabV3_spca_aug_grid_0_PS(321, 321)_BS5_EP100_LR1e-05_DS40_DR0.1_SFN32'
unet_dir = r'/hdd/Results/ct_gamma/UnetCropCV_(FixedRes)CTFinetune+nlayer9_PS(572, 572)_BS5_EP100_LR1e-05_DS50_DR0.1_SFN32'

model_name = ['deeplab', 'unet']
model_dir = [deeplab_dir, unet_dir]
gamma_list = [0.2, 0.5, 1, 1.5, 2, 2.5, 3]

plt.rcParams.update({'font.size': 14})
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
plt.savefig(os.path.join(img_dir, 'iou_vs_gamma.png'))
plt.show()
