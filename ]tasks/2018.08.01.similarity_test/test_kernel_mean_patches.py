import os
import imageio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import utils
from make_res50_features import make_res50_features
from city_building_truth import make_city_truth, make_building_truth


city_list = ['austin', 'chicago', 'kitsap', 'tyrol-w', 'vienna']
img_dir, task_dir = utils.get_task_img_folder()
model_name = 'deeplab'
feature_file_name, patch_file_name, ps, patchDir, idx = make_res50_features(model_name, task_dir, GPU=0,
                                                                            force_run=False)
file_name = os.path.join(patchDir, 'fileList.txt')
with open(file_name, 'r') as f:
    files = f.readlines()

for target_city in [1]:
    save_file_name = os.path.join(task_dir, 'target_{}_weight_loo_building.npy'.format(target_city))
    weight = np.load(save_file_name)
    weight = weight[:, 0]

    # load features and patch names
    feature = pd.read_csv(feature_file_name, sep=',', header=None).values
    with open(patch_file_name, 'r') as f:
        patch_names = f.readlines()
    sort_idx = np.argsort(weight)[::-1]
    truth_city = make_city_truth(task_dir, model_name, patch_names, force_run=False)
    truth_building = make_building_truth(ps, task_dir, model_name, patchDir, patch_names, force_run=False)
    truth_city = truth_city[np.array(idx) >= 6]
    truth_building = truth_building[np.array(idx) >= 6]
    patch_names = [patch_names[i] for i in range(len(patch_names)) if idx[i] >= 6]
    patch_names = [patch_names[i] for i in range(len(patch_names)) if
                   truth_city[i] != target_city and truth_building[i] == 1]

    remake_weight = np.zeros(np.sum(np.array(idx) >= 6) // 5 * 4)
    tb = [truth_building[i] for i in range(len(truth_building)) if truth_city[i] != target_city]
    remake_weight[np.where(tb == 1)] = weight

    print(weight.shape[0], len(patch_names), remake_weight.shape, np.sum(remake_weight))

    # plot feature weights
    plt.figure(figsize=(14, 6.5))
    for plt_cnt, patch_idx in enumerate(sort_idx[:10]):
        img = []
        for channel in range(3):
            img.append(imageio.imread(os.path.join(patchDir, '{}_RGB{}.jpg'.format(patch_names[patch_idx][:-1], channel))))
        img = np.dstack(img)

        plt.subplot(2, 5, plt_cnt + 1)
        plt.imshow(img)
        plt.axis('off')
        plt.title(patch_names[patch_idx].split('_')[0])
    plt.suptitle('Similar Patches to {}'.format(city_list[target_city]))
    plt.tight_layout()
    plt.show()
