import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
import sis_utils
import util_functions

img_dir, task_dir = sis_utils.get_task_img_folder()
csv_files = glob(os.path.join(task_dir, '*.csv'))

for file in csv_files:
    model_name = os.path.basename(file).split('_')[1]
    learn_rate = os.path.basename(file).split('_')[10][2:]
    df = pd.read_csv(os.path.join(task_dir, file), skipinitialspace=True, usecols=['Step', 'Value'])
    if model_name == 'UnetCrop':
        data = np.array(df['Value'])[:60]
        data = util_functions.savitzky_golay(data, 11, 2)
        plt.plot(np.arange(40, 100), data, label='{}: lr={}'.format(model_name, learn_rate))
    else:
        data = np.array(df['Value'])
        data = util_functions.savitzky_golay(data, 11, 2)
        plt.plot(np.arange(100), data, label='{}: lr={}'.format(model_name, learn_rate))

plt.grid('on')
plt.xlabel('Epoch')
plt.ylabel('Validation IoU')
plt.legend()
plt.savefig(os.path.join(img_dir, 'retrain_curves.png'))
plt.show()
