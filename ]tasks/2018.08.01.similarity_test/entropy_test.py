import os
import scipy.stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sis_utils
from make_res50_features import make_res50_features
from city_building_truth import make_city_truth, make_building_truth

img_dir, task_dir = sis_utils.get_task_img_folder()

model_name = 'unet'
city_list = ['austin', 'chicago', 'kitsap', 'tyrol-w', 'vienna']
entropy_list = np.zeros(len(city_list))
delta_list = np.zeros(len(city_list))

if model_name == 'unet':
    base_iou = np.array([55.7, 63.4, 56.9, 53.6, 72.6])
    mmd_iou = np.array([55.8, 64.8, 58.2, 55.7, 71.9])
else:
    base_iou = np.array([63.1, 66.3, 59.9, 54.4, 74.5])
    mmd_iou = np.array([65.2, 65.0, 61.9, 61.8, 74.5])

feature_file_name, patch_file_name, ps, patchDir, idx = make_res50_features(model_name, task_dir, GPU=0,
                                                                            force_run=False)
with open(patch_file_name, 'r') as f:
    patch_names = f.readlines()

for city_id in range(len(city_list)):
    '''weight_name = os.path.join(task_dir, '{}_loo_mmd_target_{}_5050.npy'.format(model_name, city_id))
    truth_building = make_building_truth(ps, task_dir, model_name, patchDir, patch_names, force_run=False)
    weight = np.load(weight_name)'''

    save_file_name = os.path.join(task_dir, '{}_target_{}_weight_loo_building.npy'.format(model_name, city_id))
    weight = np.load(save_file_name)
    weight = weight[:, 0]

    # load features and patch names
    '''feature = pd.read_csv(feature_file_name, sep=',', header=None).values
    with open(patch_file_name, 'r') as f:
        patch_names = f.readlines()
    sort_idx = np.argsort(weight)[::-1]
    truth_city = make_city_truth(task_dir, model_name, patch_names, force_run=False)
    truth_building = make_building_truth(ps, task_dir, model_name, patchDir, patch_names, force_run=False)
    truth_city = truth_city[np.array(idx) >= 6]
    truth_building = truth_building[np.array(idx) >= 6]
    patch_names = [patch_names[i] for i in range(len(patch_names)) if idx[i] >= 6]
    patch_names_target = [patch_names[i] for i in range(len(patch_names))
                          if truth_city[i] == city_id and truth_building[i] == 1]
    patch_names = [patch_names[i] for i in range(len(patch_names)) if
                   truth_city[i] != city_id and truth_building[i] == 1]
    weight = np.zeros(np.sum(np.array(idx) >= 6) // 5 * 4)

    print(weight.shape, len(truth_building))'''
    entropy = scipy.stats.entropy(weight)
    entropy_list[city_id] = entropy / weight.shape[0]
    delta_list[city_id] = (mmd_iou[city_id] - base_iou[city_id]) / base_iou[city_id]

X = np.arange(len(city_list))
plt.figure(figsize=(8, 6))
plt.subplot(211)
plt.plot(X, entropy_list)
plt.xticks(X, city_list)
plt.subplot(212)
plt.plot(X, delta_list)
plt.xticks(X, city_list)
plt.tight_layout()
plt.show()
