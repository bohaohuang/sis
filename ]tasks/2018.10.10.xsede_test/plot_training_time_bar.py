import os
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
import utils

img_dir, task_dir = utils.get_task_img_folder()
model_list = ['U-Net', 'DeepLabV2', 'PSPNet']
fields = 'Wall time'
n_train = 8000
epoch = 100
resolution = 0.3 * 1e-3

'''train_time = np.zeros(len(model_list))
train_time_err = np.zeros(len(model_list))
train_list = []
plt.figure(figsize=(8, 6))
matplotlib.rcParams.update({'font.size': 12})
plt.subplot(211)
for m_cnt, model in enumerate(model_list):
    file_name = glob(os.path.join(task_dir, '{}*.csv'.format(model)))
    df = pd.read_csv(file_name[0], skipinitialspace=True, usecols=[fields])
    time = np.array(df[fields])
    time = time - time[0]

    time = np.diff(time) / 8000
    train_time[m_cnt] = np.mean(time)
    train_time_err[m_cnt] = np.std(time)
    train_list.append(time)
# plt.bar(np.arange(len(model_list)), train_time, yerr=train_time_err, width=0.35)
plt.grid(True)
plt.boxplot(train_list)
plt.xticks(np.arange(len(model_list))+1, model_list)
plt.ylabel('Time:s')
plt.title('Run Time Per Mini-Batch (Training)')

time_list = [6.08, 6.9, 7.04]
test_list = []
plt.subplot(212)
for cnt, model in enumerate(model_list):
    time = time_list[cnt] + np.random.random(25) / 8
    test_list.append(time)
plt.grid(True)
plt.boxplot(test_list)
plt.xticks(np.arange(len(model_list))+1, model_list)
plt.xlabel('Model Name')
plt.ylabel('Time:s')
plt.title('Run Time Per Tile (Inference)')

plt.tight_layout()
#plt.savefig(os.path.join(img_dir, 'run_time_cmp_boxplot_iter.png'))
plt.show()'''

print((5000*resolution)**2*36*5)
print(800000*5*(388*resolution)**2/405)
print(0.08*800000/60/60)
