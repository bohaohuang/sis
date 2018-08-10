import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import utils
from make_res50_features import make_res50_features
from city_building_truth import make_city_truth


city_list = ['austin', 'chicago', 'kitsap', 'tyrol-w', 'vienna']
img_dir, task_dir = utils.get_task_img_folder()
model_name = 'deeplab'
feature_file_name, patch_file_name, ps, patchDir, idx = make_res50_features(model_name, task_dir, GPU=0,
                                                                            force_run=False)
xlim_record = [2000, 300, 1500, 1000]
ylim_record = [2, 0.3, 5, 3]

for target_city in [0, 1]:
    save_file_name = os.path.join(task_dir, 'target_{}_weight_loo.npy'.format(target_city))
    weight = np.load(save_file_name)
    weight = weight[:, 0]

    # load features and patch names
    feature = pd.read_csv(feature_file_name, sep=',', header=None).values
    with open(patch_file_name, 'r') as f:
        patch_names = f.readlines()
    truth_city = make_city_truth(task_dir, model_name, patch_names, force_run=False)

    # plot feature weights
    sort_idx = np.argsort(weight)[::-1]
    truth_city = truth_city[np.array(idx) >= 6]
    truth_city = truth_city[truth_city != target_city]
    truth_city = truth_city[sort_idx]

    plt.figure()
    for city_cnt in [a for a in range(5) if a != target_city]:
        city_ind = [a for a in np.arange(weight.shape[0]) if truth_city[a] == city_cnt]
        plt.bar(city_ind, weight[sort_idx[city_ind]], label=city_list[city_cnt])
    plt.legend()
    plt.xlim([0, xlim_record[target_city]])
    plt.ylim([0, ylim_record[target_city]])
    plt.tight_layout()
    plt.title('Target {} / Source 5 Cites (>26)'.format(city_list[target_city]))
    # plt.savefig(os.path.join(img_dir, 'target_{}_source_5g26.png'.format(target_city)))
plt.show()
