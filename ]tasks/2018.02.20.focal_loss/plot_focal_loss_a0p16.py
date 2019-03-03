import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from math import factorial
import sis_utils

def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order + 1)
    half_window = (window_size - 1) // 2
    # precompute coefficients
    b = np.mat([[k ** i for i in order_range] for k in range(-half_window, half_window + 1)])
    m = np.linalg.pinv(b).A[deriv] * rate ** deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve(m[::-1], y, mode='valid')

img_dir, task_dir = sis_utils.get_task_img_folder()
files = glob(os.path.join(task_dir, '*.csv'))
fields = ['Step', 'Value']

plt.rcParams.update({'font.size': 16})
plt.figure(figsize=(8, 4))
for file in files:
    df = pd.read_csv(file, skipinitialspace=True, usecols=fields)
    value = savitzky_golay(np.array(df['Value']), 11, 2)
    label_str = ''
    if 'UnetCrop' in file and '_xent_' in file:
        print(file)
        color_str = 'b'
        lst = '--'
        label_str += 'Unet xent'
    elif 'Unet' and '0.84' in file:
        color_str = 'g'
        lst = '-'
        gamma = float(file.split('_g')[1][0])
        label_str += 'Unet gamma:{}'.format(gamma)
    else:
        continue
    plt.plot(np.arange(100), value, linestyle=lst, color=color_str, label=label_str)
plt.xlabel('Epoch')
plt.ylabel('IoU')
plt.ylim([0.5, 0.78])
plt.legend()
plt.grid('on')
plt.title('Validation IoU in Training')
plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'focal_cmp2.png'))
plt.show()
