import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import utils


def sliding_mean(data_array, window=5):
    new_list = []
    for i in range(data_array.shape[0]):
        indices = range(max(i-window+1, 0), min(i+window+1, data_array.shape[0]))
        avg = 0
        for j in indices:
            avg += data_array[j]
            avg /= float(len(indices))
            new_list.append(avg)
    return np.array(new_list)


def exponential_smoothing(series, alpha=0.6):
    result = [series[0]] # first value is same as series
    for n in range(1, series.shape[0]):
        result.append(alpha * series[n] + (1 - alpha) * result[n-1])
    return np.array(result)


img_dir, task_dir = utils.get_task_img_folder()
file_names = ['run_UnetCrop_inria_aug_incity_PS(572, 572)_BS5_EP100_LR0.0001_DS60_DR0.1_SFN32-tag-xent_validation.csv',
              'run_exp1_UnetCrop_inria_aug_grid_0_PS(572, 572)_BS5_EP100_LR0.0001_DS60_DR0.1_SFN32-tag-xent_validation.csv',
              'run_UnetCrop_inria_aug_xcity_PS(572, 572)_BS5_EP100_LR0.0001_DS60_DR0.1_SFN32-tag-xent_validation.csv']
fields = ['Step', 'Value']
run_type = ['in city (IoU=54.05)', 'grid (IoU=72.70)', 'x city (IoU=73.55)']

plt.figure(figsize=(8, 4))
for cnt, file in enumerate(file_names):
    df = pd.read_csv(os.path.join(task_dir, file), skipinitialspace=True, usecols=fields)
    step = exponential_smoothing(np.array(df['Step']))
    value = exponential_smoothing(np.array(df['Value']))

    plt.plot(step/1600, value, label=run_type[cnt])
plt.legend()
plt.xlabel('Epoch Number')
plt.ylabel('Cross Entropy')
plt.title('Mini Batch Scheme Comparison')
plt.savefig(os.path.join(img_dir, 'exp2_cmp.png'))
plt.show()
