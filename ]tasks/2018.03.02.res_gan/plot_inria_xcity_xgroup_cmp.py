import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

city_dir = r'/hdd/Results/xcity_vs_incity'
group_dir = r'/hdd/Results/xgroup'
incity_dir = glob(os.path.join(city_dir, 'DeeplabV3_inria_aug_incity*/'))
xcity_dir = glob(os.path.join(city_dir, 'DeeplabV3_inria_aug_xcity*/'))
xgroup_dir = glob(os.path.join(group_dir, 'DeeplabV3_inria_aug_train_fileList*/'))

incity_record = np.zeros(len(incity_dir))
xcity_record = np.zeros(len(xcity_dir))
xgroup_record = np.zeros(len(xgroup_dir))

for cnt, fold in enumerate(incity_dir):
    result_file = os.path.join(fold, 'inria', 'result.txt')
    with open(result_file, 'r') as f:
        lines = f.readlines()
    incity_record[cnt] = float(lines[-1])

for cnt, fold in enumerate(xcity_dir):
    result_file = os.path.join(fold, 'inria', 'result.txt')
    with open(result_file, 'r') as f:
        lines = f.readlines()
    xcity_record[cnt] = float(lines[-1])

for cnt, fold in enumerate(xgroup_dir):
    result_file = os.path.join(fold, 'inria', 'result.txt')
    with open(result_file, 'r') as f:
        lines = f.readlines()
    xgroup_record[cnt] = float(lines[-1])

plt.boxplot([incity_record, xcity_record, xgroup_record])
plt.show()
