import os
import numpy as np
import matplotlib.pyplot as plt
import utils

loo_dir_deeplab = r'/hdd/Results/DeeplabV3_inria_aug_leave_0_0_PS(321, 321)_BS5_EP100_LR1e-05_DS40_DR0.1_SFN32/default'
loo_dir_deeplab_finetune = r'/hdd/Results/DeeplabV3_inria_aug_leave_0_0_0_PS(321, 321)_BS5_EP20_LR1e-07_DS10_DR0.1_SFN32/default'
loo_dir_deeplab_auto = r'/hdd/Results/DeeplabV3_inria_aug_leave_auto_0_0_PS(321, 321)_BS5_EP20_LR1e-07_DS40_DR0.1_SFN32/default'
pred_dir_xr_deeplab = r'/hdd/Results/DeeplabV3_res101_inria_aug_grid_0_PS(321, 321)_BS5_EP100_LR1e-05_DS40_DR0.1_SFN32/default'

scores = np.zeros((4, 6))

for cnt_1, loo_dir in enumerate([loo_dir_deeplab, loo_dir_deeplab_finetune, loo_dir_deeplab_auto, pred_dir_xr_deeplab]):
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
for loo_dir in [loo_dir_deeplab, loo_dir_deeplab_finetune, loo_dir_deeplab_auto, pred_dir_xr_deeplab]:
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

img_dir, task_dir = utils.get_task_img_folder()
ind = np.arange(6)
width = 0.2
scores *= 100

plt.figure(figsize=(8, 6))
plt.rcParams.update({'font.size': 12})
plt.bar(ind, scores[0, :], width=width, label='Leave-one-out')
plt.bar(ind+width, scores[1, :], width=width, label='Finetune')
plt.bar(ind+width*2, scores[2, :], width=width, label='Auto')
plt.bar(ind+width*3, scores[3, :], width=width, label='Cross Region')
'''plt.axhline(y=baseline[0], linestyle='--', color='blue')
plt.axhline(y=baseline[1], linestyle='--', color='darkorange')
plt.axhline(y=baseline[2], linestyle='--', color='green')
plt.axhline(y=baseline[3], linestyle='--', color='red')'''
plt.xticks(ind+width*1.5, ['Austin1', 'Austin2', 'Austin3', 'Austin4', 'Austin5', 'Avg'])
plt.ylabel('IoU')
plt.title('Leave-one-out Vs Finetune')
plt.legend(loc='upper left', ncol=4)
plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'loo_cmp_austin_auto.png'))
plt.show()
