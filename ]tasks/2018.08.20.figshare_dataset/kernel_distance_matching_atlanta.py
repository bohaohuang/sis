"""
Do kernel matching on Res50 features
The matching function comes from https://github.com/vodp/py-kmm/blob/master/Kernel%20Meam%20Matching.ipynb
"""
import csv
import math
import imageio
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from cvxopt import matrix, solvers, spmatrix
from city_building_truth import make_building_truth, make_city_truth
from gmm_cluster import *
from make_res50_features import crop_center
from make_res50_features import make_res50_features as mrf
import uab_collectionFunctions
import uab_DataHandlerFunctions
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


def make_res50_features(model_name, task_dir, GPU=0, force_run=False):
    tf.reset_default_graph()
    feature_file_name = os.path.join(task_dir, 'res50_atlanta_{}.csv'.format(model_name))
    patch_file_name = os.path.join(task_dir, 'res50_atlanta_{}.txt'.format(model_name))

    if model_name == 'deeplab':
        input_size = (321, 321)
        overlap = 0
    else:
        input_size = (572, 572)
        overlap = 184
    blCol = uab_collectionFunctions.uabCollection('atlanta')
    img_mean = blCol.getChannelMeans([0, 1, 2])
    extrObj = uab_DataHandlerFunctions.uabPatchExtr([0, 1, 2, 3],
                                                    cSize=input_size,
                                                    numPixOverlap=overlap,
                                                    extSave=['jpg', 'jpg', 'jpg', 'png'],
                                                    isTrain=True,
                                                    gtInd=3,
                                                    pad=overlap // 2)
    patchDir = extrObj.run(blCol)

    if not os.path.exists(feature_file_name) or not os.path.exists(patch_file_name) or force_run:
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(GPU)
        import keras

        input_size_fit = (224, 224)

        file_name = os.path.join(patchDir, 'fileList.txt')
        with open(file_name, 'r') as f:
            files = f.readlines()

        res50 = keras.applications.resnet50.ResNet50(include_top=True, weights='imagenet')
        fc2048 = keras.models.Model(inputs=res50.input, outputs=res50.get_layer('flatten_1').output)
        with open(feature_file_name, 'w+') as f:
            with open(patch_file_name, 'w+') as f2:
                for file_line in tqdm(files):
                    patch_name = file_line.split('.')[0][:-5]
                    img = np.zeros((input_size[0], input_size[1], 3), dtype=np.uint8)
                    for cnt, file in enumerate(file_line.strip().split(' ')[:3]):
                        img[:, :, cnt] = imageio.imread(os.path.join(patchDir, file)) - img_mean[cnt]

                    img = np.expand_dims(crop_center(img, input_size_fit[0], input_size_fit[1]), axis=0)

                    fc1000 = fc2048.predict(img).reshape((-1,)).tolist()
                    writer = csv.writer(f, lineterminator='\n')
                    writer.writerow(['{}'.format(x) for x in fc1000])
                    f2.write('{}\n'.format(patch_name))

    return feature_file_name, patch_file_name, input_size[0], patchDir


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


model_name = 'unet'
perplex = 25
top_cnt = 5
target_city = 'dc'
force_run = False

# 1. make features
img_dir, task_dir = utils.get_task_img_folder()
feature_file_name, patch_file_name, ps, patchDir, idx = mrf(model_name, task_dir, GPU=1, force_run=False)
feature = pd.read_csv(feature_file_name, sep=',', header=None).values
with open(patch_file_name, 'r') as f:
    patch_names = f.readlines()
target_feature_file_name, _, _, _ = make_res50_features(model_name, task_dir, GPU=1, force_run=True)
target_feature = pd.read_csv(target_feature_file_name, sep=',', header=None).values

# 2. make city and building truth
truth_city = make_city_truth(task_dir, model_name, patch_names, force_run=False)
truth_building = make_building_truth(ps, task_dir, model_name, patchDir, patch_names, force_run=False)

# 3. do feature mapping
# mean_dist = compute_median_distance(source_feature)
# print(mean_dist)
source_feature = select_feature(feature, np.array(idx) >= 6, truth_city, truth_building,
                                [i for i in range(5)], True)
match_file_name = os.path.join(task_dir, '{}_match_{}_top{}.npy'.format(model_name, target_city, top_cnt))
dist_file_name = os.path.join(task_dir, '{}_dist_{}_top{}.npy'.format(model_name, target_city, top_cnt))
if not os.path.exists(match_file_name) or not os.path.exists(dist_file_name) or force_run:
    match_record, dist_record = distance_matching(source_feature, target_feature, top_cnt=top_cnt)
    np.save(match_file_name, match_record)
    np.save(dist_file_name, dist_record)
else:
    match_record = np.load(match_file_name)
    dist_record = np.load(dist_file_name)

sample_prior = softmax(-dist_record, t=50)
remake_weight = np.zeros(np.sum(np.array(idx) >= 6))
tb = [truth_building[i] for i in range(len(truth_building)) if truth_city[i] != target_city]
weight_cnt = 0
for i in range(remake_weight.shape[0]):
    if tb[i] == 1:
        remake_weight[i] = sample_prior[weight_cnt]
        weight_cnt += 1
remake_weight = remake_weight / np.sum(remake_weight) / 2
remake_weight[remake_weight == 0] = 1 / 2 / np.sum(remake_weight == 0)
remake_weight = remake_weight / np.sum(remake_weight)
print(np.sum(remake_weight))
np.save(os.path.join(task_dir, '{}_loo_distance_target_{}_5050.npy'.format(model_name, target_city)), remake_weight)

plt.bar(np.arange(len(remake_weight)), np.sort(remake_weight)[::-1])
plt.show()

patch_names = [patch_names[a] for a in range(len(idx)) if idx[a] >= 6]
sort_idx = np.argsort(remake_weight)[::-1]
cnt = 0
while True:
    plt.figure(figsize=(12, 4))
    for i in range(5):
        img = []
        for channel in range(3):
            img.append(imageio.imread(os.path.join(patchDir, '{}_RGB{}.jpg'.format(patch_names[sort_idx[cnt]][:-1], channel))))
        img = np.dstack(img)

        gt = imageio.imread(os.path.join(patchDir, '{}_GT_Divide.png'.format(patch_names[sort_idx[cnt]][:-1])))
        plt.subplot(2, 5, 1 + i)
        plt.imshow(img)
        plt.axis('off')
        plt.subplot(2, 5, 6 + i)
        plt.imshow(gt)
        plt.axis('off')
        cnt += 1
    plt.tight_layout()
    plt.show()
