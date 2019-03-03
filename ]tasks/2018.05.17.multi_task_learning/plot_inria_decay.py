import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import sis_utils
import util_functions


img_dir, task_dir = sis_utils.get_task_img_folder()

# plot iou curves
'''results = np.zeros((2, 4))
for cnt_1, ep in enumerate([100, 200]):
    for cnt_2, (DS, DR) in enumerate(zip([2.0, 5.0, 40.0, 60.0], [0.9, 0.9, 0.1, 0.1])):
        model_dir = r'/hdd/Results/inria_decay/UnetCrop_inria_decay_0_PS(572, 572)_BS5_EP{}_LR0.0001_DS{}_DR{}_SFN32'.\
            format(ep, DS, DR)
        model_result = os.path.join(model_dir, 'inria', 'result.txt')

        city_iou_a = np.zeros(6)
        city_iou_b = np.zeros(6)

        with open(model_result, 'r') as f:
            result_record = f.readlines()
        for cnt, line in enumerate(result_record[:-1]):
            A, B = line.split('(')[1].strip().strip(')').split(',')
            city_iou_a[cnt // 5] += float(A)
            city_iou_b[cnt // 5] += float(B)
        city_iou_a[-1] = np.sum(city_iou_a[:-1])
        city_iou_b[-1] = np.sum(city_iou_b[:-1])
        city_iou = city_iou_a / city_iou_b * 100

        results[cnt_1, cnt_2] = city_iou[-1]

plt.figure(figsize=(8, 4))
plt.plot(results[0, :], marker='o', label='EP=100')
plt.plot(results[1, :], marker='o', label='EP=200')
plt.xticks(np.arange(4), ['DS=2\nDR=0.9', 'DS=5\nDR=0.9', 'DS=40\nDR=0.1', 'DS=60\nDR=0.1'])
plt.xlabel('Decay Step & Rate')
plt.ylabel('IoU')
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(img_dir, 'decay_iou_cmp.png'))
plt.show()'''

# plot iou and lr training curves
fields = ['Step', 'Value']
lsty = ['-', '--']
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
plt.figure(figsize=(10, 8))
for cnt_1, ep in enumerate([100, 200]):
    for cnt_2, (DS, DR) in enumerate(zip([2.0, 5.0, 40.0, 60.0], [0.9, 0.9, 0.1, 0.1])):
        iou_file_name = r'run_UnetCrop_inria_decay_0_PS(572, 572)_BS5_EP{}_LR0.0001_DS{}_DR{}_SFN32-tag-iou_validation.csv'.\
            format(ep, DS, DR)
        lr_file_name = r'run_UnetCrop_inria_decay_0_PS(572, 572)_BS5_EP{}_LR0.0001_DS{}_DR{}_SFN32-tag-learning_rate.csv'.\
            format(ep, DS, DR)

        df = pd.read_csv(os.path.join(task_dir, iou_file_name), skipinitialspace=True, usecols=fields)
        value = util_functions.savitzky_golay(np.array(df['Value']), 11, 2)
        plt.subplot(211)
        plt.plot(value * 100, label='EP:{} DS:{} DR:{}'.format(ep, DS, DR), linestyle=lsty[cnt_1], color=colors[cnt_2])

        df = pd.read_csv(os.path.join(task_dir, lr_file_name), skipinitialspace=True, usecols=fields)
        value = df['Value']
        step = df['Step']
        plt.subplot(212)
        plt.plot(step/1600, value, label='EP:{} DS:{} DR:{}'.format(ep, DS, DR), linestyle=lsty[cnt_1], color=colors[cnt_2])
plt.subplot(211)
plt.grid('on')
plt.ylim([65, 80])
plt.ylabel('IoU')
ax = plt.subplot(212)
plt.grid('on')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Learn Rate')
ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'decay_train_curve_cmp.png'))
plt.show()
