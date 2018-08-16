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
model_name = 'unet'
feature_file_name, patch_file_name, ps, patchDir, idx = make_res50_features(model_name, task_dir, GPU=0,
                                                                            force_run=False)
file_name = os.path.join(patchDir, 'fileList.txt')
with open(file_name, 'r') as f:
    files = f.readlines()

for target_city in [2, 3]:
    save_file_name = os.path.join(task_dir, '{}_target_{}_weight_xregion_building.npy'.format(model_name, target_city))
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
    patch_names_target = [patch_names[i] for i in range(len(patch_names))
                          if truth_building[i] == 1]
    patch_names = [patch_names[i] for i in range(len(patch_names)) if
                   truth_building[i] == 1]

    remake_weight = np.zeros(np.sum(np.array(idx) >= 6))
    tb = [truth_building[i] for i in range(len(truth_building))]
    weight_cnt = 0
    for i in range(remake_weight.shape[0]):
        if tb[i] == 1:
            remake_weight[i] = weight[weight_cnt]
            weight_cnt += 1
    remake_weight = remake_weight / np.sum(remake_weight) / 2
    remake_weight[remake_weight == 0] = 1 / 2 / np.sum(remake_weight == 0)

    remake_weight = remake_weight / np.sum(remake_weight)
    np.save(os.path.join(task_dir, '{}_xregion_mmd_target_{}_5050.npy'.format(model_name, target_city)), remake_weight)

    # plot feature weights
    '''plt.figure(figsize=(16, 5.5))
    for plt_cnt, patch_name in enumerate(np.random.permutation(patch_names_target)[:2]):
        img = []
        for channel in range(3):
            img.append(imageio.imread(os.path.join(patchDir, '{}_RGB{}.jpg'.format(patch_name[:-1], channel))))
        img = np.dstack(img)

        plt.subplot(2, 6, plt_cnt * 6 + 1)
        plt.imshow(img)
        plt.axis('off')
        plt.title(patch_name.split('_')[0])
    for plt_cnt, patch_idx in enumerate(sort_idx[:10]):
        img = []
        for channel in range(3):
            img.append(imageio.imread(os.path.join(patchDir, '{}_RGB{}.jpg'.format(patch_names[patch_idx][:-1], channel))))
        img = np.dstack(img)

        plt.subplot(2, 6, plt_cnt // 5 * 6 + plt_cnt % 5 + 2)
        plt.imshow(img)
        plt.axis('off')
        plt.title(patch_names[patch_idx].split('_')[0])
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, 'similar_patches_{}_unet'.format(city_list[target_city])))
    plt.show()'''
