"""
Do kernel matching on Res50 features
The matching function comes from https://github.com/vodp/py-kmm/blob/master/Kernel%20Meam%20Matching.ipynb
"""
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
from cvxopt import matrix, solvers
from make_res50_features import make_res50_features
from city_building_truth import make_building_truth, make_city_truth
from gmm_cluster import  *
import utils


def kernel_mean_matching(X, Z, kern='lin', B=1.0, sigma=1.0, eps=None):
    nx = X.shape[0]
    nz = Z.shape[0]
    if eps == None:
        eps = B / math.sqrt(nz)
    if kern == 'lin':
        K = np.dot(Z, Z.T)
        kappa = np.sum(np.dot(Z, X.T) * float(nz) / float(nx), axis=1)
    elif kern == 'rbf':
        K = compute_rbf(Z, Z, sigma=sigma)
        kappa = np.sum(compute_rbf(Z, X, sigma=sigma), axis=1) * float(nz) / float(nx)
    else:
        raise ValueError('unknown kernel')

    K = matrix(K)
    kappa = matrix(kappa)
    G = matrix(np.r_[np.ones((1, nz)), -np.ones((1, nz)), np.eye(nz), -np.eye(nz)])
    h = matrix(np.r_[nz * (1 + eps), nz * (eps - 1), B * np.ones((nz,)), np.zeros((nz,))])

    sol = solvers.qp(K, -kappa, G, h)
    coef = np.array(sol['x'])
    return coef


def compute_rbf(X, Z, sigma=1.0):
    K = np.zeros((X.shape[0], Z.shape[0]), dtype=float)
    for i, vx in enumerate(X):
        K[i, :] = np.exp(-np.sum((vx - Z) ** 2, axis=1) / (2.0 * sigma))
    return K


def compute_median_distance(X):
    n_sample = X.shape[0]
    dist = 0
    for i in tqdm(range(0, n_sample, 20)):
        dist += np.sum(np.linalg.norm(X[(i+1):, :] - X[i, :], axis=1)) / (n_sample * (n_sample - 1))
    return dist


model_name = 'deeplab'
perplex = 25

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
source_feature = select_feature(feature, np.array(idx) >= 6, truth_city, truth_building, range(5), False)
# mean_dist = compute_median_distance(source_feature)
# print(mean_dist)
for target_city in tqdm(range(5)):
    target_feature = select_feature(feature, np.array(idx) < 6, truth_city, truth_building, [target_city], False)
    weight = kernel_mean_matching(target_feature, source_feature, kern='rbf', B=1000.0, sigma=21.16)
    save_file_name = os.path.join(task_dir, 'target_{}_weight.npy'.format(target_city))
    np.save(save_file_name, weight)
