"""
Do kernel distance on Res50 features
"""
import imageio
import numpy as np
import pandas as pd
from tqdm import tqdm
from make_res50_features import make_res50_features
from city_building_truth import make_building_truth, make_city_truth
from gmm_cluster import  *
import utils


def compute_distance(m, f):
    return np.linalg.norm((m-f), axis=1)


def distance_matching(f_s, f_t, top_cnt=5):
    n, _ = f_s.shape
    match_record = np.zeros((n, top_cnt), dtype=np.uint32)
    dist_record = np.zeros(n)
    for i in range(n):
        dist = compute_distance(f_t, f_s[i, :])
        match_record[i, :] = np.argsort(dist)[:top_cnt]
        dist_record[i] = np.mean(dist[match_record[i, :]])
    return match_record, dist_record


model_name = 'deeplab'
top_cnt = 5
force_run = False

for target_city in tqdm(range(5)):
    # 1. make features
    img_dir, task_dir = utils.get_task_img_folder()
    feature_file_name, patch_file_name, ps, patchDir, idx = make_res50_features(model_name, task_dir, GPU=0,
                                                                                force_run=False)
    feature = pd.read_csv(feature_file_name, sep=',', header=None).values
    with open(patch_file_name, 'r') as f:
        patch_names = f.readlines()

    # 2. make city and building truth
    truth_city = make_city_truth(task_dir, model_name, patch_names, force_run=False)
    truth_building = make_building_truth(ps, task_dir, model_name, patchDir, patch_names, force_run=False)

    # 3. do feature mapping
    source_feature = select_feature(feature, np.array(idx) >= 6, truth_city, truth_building,
                                    [i for i in range(5)], True)
    target_feature = select_feature(feature, np.array(idx) < 6, truth_city, truth_building, [target_city], False)

    match_file_name = os.path.join(task_dir, '{}_match_{}_top{}_xregion.npy'.format(model_name, target_city, top_cnt))
    dist_file_name = os.path.join(task_dir, '{}_dist_{}_top{}_xregion.npy'.format(model_name, target_city, top_cnt))
    if not os.path.exists(match_file_name) or not os.path.exists(dist_file_name) or force_run:
        match_record, dist_record = distance_matching(source_feature, target_feature, top_cnt=top_cnt)
        np.save(match_file_name, match_record)
        np.save(dist_file_name, dist_record)
    else:
        match_record = np.load(match_file_name)
        dist_record = np.load(dist_file_name)

    with open(patch_file_name, 'r') as f:
        patch_names = f.readlines()
    sort_idx = np.argsort(dist_record)
    truth_city = make_city_truth(task_dir, model_name, patch_names, force_run=False)
    truth_building = make_building_truth(ps, task_dir, model_name, patchDir, patch_names, force_run=False)
    truth_city_target = truth_city[np.array(idx) < 6]
    truth_city = truth_city[np.array(idx) >= 6]
    truth_building = truth_building[np.array(idx) >= 6]
    patch_names_target = [patch_names[i] for i in range(len(patch_names)) if idx[i] < 6]
    patch_names = [patch_names[i] for i in range(len(patch_names)) if idx[i] >= 6]
    patch_names_target = [patch_names_target[i] for i in range(len(patch_names_target))
                          if truth_city_target[i] == target_city]
    patch_names = [patch_names[i] for i in range(len(patch_names)) if
                   truth_city[i] != target_city and truth_building[i] == 1]

    sample_prior = softmax(-dist_record, t=100)
    remake_weight = np.zeros(np.sum(np.array(idx) >= 6))
    tb = [truth_building[i] for i in range(len(truth_building))]
    weight_cnt = 0
    for i in range(remake_weight.shape[0]):
        if tb[i] == 1:
            remake_weight[i] = sample_prior[weight_cnt]
            weight_cnt += 1
    remake_weight = remake_weight / np.sum(remake_weight) / 2
    remake_weight[remake_weight == 0] = 1 / 2 / np.sum(remake_weight == 0)
    remake_weight = remake_weight / np.sum(remake_weight)
    print(np.sum(remake_weight))
    np.save(os.path.join(task_dir, '{}_xregion_distance_target_{}_5050.npy'.format(model_name, target_city)), remake_weight)
