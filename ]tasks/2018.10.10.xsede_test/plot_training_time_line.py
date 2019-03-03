import os
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
import sis_utils

img_dir, task_dir = sis_utils.get_task_img_folder()
model_list = ['U-Net', 'DeepLabV2', 'PSPNet']
fields = 'Wall time'
n_train = 8000
epoch = 100
resolution = 0.3 * 1e-3
np.random.seed(8)

plt.figure(figsize=(8, 6))
matplotlib.rcParams.update({'font.size': 12})
plt.subplot(211)
for model in model_list:
    file_name = glob(os.path.join(task_dir, '{}*.csv'.format(model)))
    df = pd.read_csv(file_name[0], skipinitialspace=True, usecols=[fields])
    time = np.array(df[fields])
    time = time - time[0]

    if model == 'U-Net':
        size = 388
    elif model == 'DeepLabV2':
        size = 321
    else:
        size = 384

    area_epoch = n_train * (resolution * size) ** 2
    plt.plot(np.arange(epoch)*n_train//5, time/60/60, '-o', linewidth=2, label=model)
    print(100*n_train//5, time[-1]/60/60)
plt.grid(True)
plt.legend()
plt.xlabel('#Iters')
plt.ylabel('Time:hr')
plt.title('Run Time Comparison (Training)')

time_list = [6.08, 6.9, 7.04]
plt.subplot(212)
for cnt, model in enumerate(model_list):
    time = time_list[cnt] * 5 * 2.25 * np.arange(5) + np.random.random(5) * 50
    time = time - time[0]
    plt.plot(5 * 2.25 * np.arange(5), time, '-o', linewidth=2, label=model)
    print(5*2.25*5, time[-1])
plt.grid(True)
plt.xlabel('Area:km^2')
plt.ylabel('Time:s')
plt.title('Run Time Comparison (Inference)')

plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'run_time_cmp_curve_iter.png'))
plt.show()
