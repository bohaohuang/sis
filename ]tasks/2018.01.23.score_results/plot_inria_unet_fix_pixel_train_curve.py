import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import factorial
import utils


def exponential_smoothing(series, alpha=0.6):
    result = [series[0]] # first value is same as series
    for n in range(1, series.shape[0]):
        result.append(alpha * series[n] + (1 - alpha) * result[n-1])
    return np.array(result)


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


batch_sizes = [10,  9,   8,   7,   6,   5,   4,   3,   2,   1]
patch_sizes = [460, 476, 492, 508, 540, 572, 620, 684, 796, 1052]
fields = ['Step', 'Value']

img_dir, task_dir = utils.get_task_img_folder()
run_record_dir = os.path.join(task_dir, 'unet_inria_fixel_pixel')

plt.rcParams.update({'font.size': 14})
plt.figure(figsize=(12, 5))

for batch_size, patch_size in zip(batch_sizes, patch_sizes):
    records = np.zeros((5, 100))
    for runId in range(5):
        file_name = 'run_UnetCrop_inria_aug_psbs_{}_PS({}, {})_BS{}_EP100_LR0.0001_DS60_DR0.1_SFN32' \
                    '-tag-xent_validation.csv'.format(runId, patch_size, patch_size, batch_size)

        df = pd.read_csv(os.path.join(run_record_dir, file_name), skipinitialspace=True, usecols=fields)
        value = savitzky_golay(np.array(df['Value']), 11, 2)
        records[runId, :] = value[:100]
    records = np.mean(records, axis=0)

    plt.plot(np.arange(100), records, linewidth=2, label='PS:{:4}'.format(patch_size))
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Xent')
plt.title('Unet on Inria Training Curve')
plt.tight_layout()

plt.savefig(os.path.join(img_dir, 'unet_inria_fixpixel_train_curve.png'))

plt.show()
