import os
import csv
import imageio
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.manifold import TSNE
import utils

run_clustering = False
img_dir, task_dir = utils.get_task_img_folder()
npy_file_name = os.path.join(task_dir, 'encoded_res50_inria_spca.npy')
city_list = ['inria', 'sp']
np.random.seed(1004)

if run_clustering:
    file_name = os.path.join(task_dir, 'res50_fc1000_inria.csv')
    features = []
    with open(file_name, 'r') as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            features.append(row)
    len1 = len(features)

    file_name = os.path.join(task_dir, 'temp', 'res50_fc1000_sp_deeplab.csv')
    with open(file_name, 'r') as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            features.append(row)
    len2 = len(features) - len1

    feature_encode = TSNE(n_components=2, perplexity=25, verbose=True).fit_transform(features)
    np.save(npy_file_name, [feature_encode, len1, len2])
else:
    feature_encode, len1, len2 = np.load(npy_file_name)
feature_encode = np.array(feature_encode)

random_idx = np.random.binomial(1, 0.2, feature_encode.shape[0])
patch_ids = np.arange(feature_encode.shape[0])
feature_encode = feature_encode[random_idx == 1, :]
patch_ids = patch_ids[random_idx == 1]
patch_group = np.concatenate((np.zeros(len1), np.ones(len2)))
patch_group = patch_group[random_idx == 1]

cmap = plt.get_cmap('Set1').colors
patch_id = np.arange(feature_encode.shape[0])
plt.figure(figsize=(15, 8))
for i in range(2):
    plt.scatter(feature_encode[patch_group == i, 0], feature_encode[patch_group == i, 1], color=cmap[i],
                label=city_list[i], edgecolors='k')
#plt.scatter(feature_encode[:, 0], feature_encode[:, 1], c=patch_percent, cmap=plt.get_cmap('bwr'))
#plt.colorbar()

for i in range(feature_encode.shape[0]):
    plt.text(feature_encode[i, 0], feature_encode[i, 1], patch_ids[i])
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('TSNE Projection Result')
plt.legend()
plt.tight_layout()
#plt.savefig(os.path.join(img_dir, 'res50_tsne_proj_inria_spca.png'))
plt.show()
