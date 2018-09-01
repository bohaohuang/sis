import os
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt
import utils

img_dir, task_dir = utils.get_task_img_folder()

model_name = 'deeplab'
city_list = ['austin', 'chicago', 'kitsap', 'tyrol-w', 'vienna']
entropy_list = np.zeros(len(city_list))
delta_list = np.zeros(len(city_list))

if model_name == 'unet':
    base_iou = np.array([55.7, 63.4, 56.9, 53.6, 72.6])
    mmd_iou = np.array([55.8, 64.8, 58.2, 55.7, 71.9])
else:
    base_iou = np.array([63.1, 66.3, 59.9, 54.4, 74.5])
    mmd_iou = np.array([65.2, 65.0, 61.9, 61.8, 74.5])

for city_id in range(len(city_list)):
    weight_name = os.path.join(task_dir, '{}_loo_mmd_target_{}_5050.npy'.format(model_name, city_id))
    weight = np.load(weight_name)
    entropy = scipy.stats.entropy(weight)
    entropy_list[city_id] = entropy
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
