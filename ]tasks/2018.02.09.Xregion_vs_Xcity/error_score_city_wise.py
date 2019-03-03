import os
import numpy as np
import matplotlib.pyplot as plt
import sis_utils

loo_dir_deeplab = r'/hdd/Results/DeeplabV3_inria_aug_leave_0_0_PS(321, 321)_BS5_EP100_LR1e-05_DS40_DR0.1_SFN32/default'
loo_dir_unet = r'/hdd/Results/UnetCrop_inria_aug_leave_0_0_PS(572, 572)_BS5_EP100_LR0.0001_DS60_DR0.1_SFN32/default'
pred_dir_xr_unet = r'/hdd/Results/grid_vs_random/UnetCrop_inria_aug_grid_0_PS(572, 572)_BS5_EP100_LR0.0001_DS60_DR0.1_SFN32/inria'
pred_dir_xr_deeplab = r'/hdd/Results/DeeplabV3_res101_inria_aug_grid_0_PS(321, 321)_BS5_EP100_LR1e-05_DS40_DR0.1_SFN32/default'

scores = np.zeros((2, 6))

for cnt_1, loo_dir in enumerate([loo_dir_deeplab, loo_dir_unet]):
    file_name = os.path.join(loo_dir, 'result.txt')
    with open(file_name, 'r') as f:
        lines = f.readlines()
    A_record = 0
    B_record = 0
    austin = lines[:5]
    for cnt_2, item in enumerate(austin):
        A = int(item.split('(')[1].split(',')[0])
        B = int(item.split(' ')[-1].split(')')[0])
        A_record += A
        B_record += B
        scores[cnt_1, cnt_2] = A/B
    scores[cnt_1, -1] = A_record/B_record

baseline = []
for loo_dir in [pred_dir_xr_deeplab, pred_dir_xr_unet]:
    file_name = os.path.join(loo_dir, 'result.txt')
    with open(file_name, 'r') as f:
        lines = f.readlines()
    A_record = 0
    B_record = 0
    austin = lines[:5]
    for cnt_2, item in enumerate(austin):
        A = int(item.split('(')[1].split(',')[0])
        B = int(item.split(' ')[-1].split(')')[0])
        A_record += A
        B_record += B
    baseline.append(A_record/B_record*100)

img_dir, task_dir = sis_utils.get_task_img_folder()
ind = np.arange(6)
width = 0.35
scores *= 100

plt.rcParams.update({'font.size': 12})
plt.bar(ind, scores[0, :], width=width, label='DeepLab')
plt.bar(ind+width, scores[1, :], width=width, label='U-Net')
plt.axhline(y=baseline[0], color='g', linestyle='--')
plt.axhline(y=baseline[1], color='g', linestyle='--')
plt.xticks(ind+width/2, ['Austin1', 'Austin2', 'Austin3', 'Austin4', 'Austin5', 'Avg'])
plt.ylabel('IoU')
plt.title('Leave-one-out Performance Comparison')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'loo_cmp_austin.png'))
plt.show()
