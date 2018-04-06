import os
import csv
import utils
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

run_clustering = False
img_dir, task_dir = utils.get_task_img_folder()
npy_file_name = os.path.join(task_dir, 'encoded_res50_2.npy')
city_order = {'aus': 0, 'chi': 1, 'kit': 2, 'tyr': 3, 'vie': 4}
city_list = ['austin', 'chicago', 'kitsap', 'tyrol-w', 'vienna']
np.random.seed(1004)

if run_clustering:
    file_name = os.path.join(task_dir, 'res50_fc1000_inria.csv')
    features = []
    with open(file_name, 'r') as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            features.append(row)

    feature_encode = TSNE(n_components=2, perplexity=25, verbose=True).fit_transform(features)
    np.save(npy_file_name, feature_encode)
else:
    feature_encode = np.load(npy_file_name)
feature_encode = np.array(feature_encode)

random_idx = np.random.binomial(1, 0.2, feature_encode.shape[0])
patch_ids = np.arange(feature_encode.shape[0])
feature_encode = feature_encode[random_idx == 1, :]
patch_ids = patch_ids[random_idx == 1]

patch_name_fname = os.path.join(task_dir, 'res50_fc1000_inria.txt')
with open(patch_name_fname, 'r') as f:
    patch_name_list = f.readlines()
patch_name_list = [city_order[a[:3]] for a in patch_name_list]

patch_name_code = []
for i in patch_name_list:
    patch_name_code.append(i)
patch_name_code = np.array(patch_name_code)
patch_name_code = patch_name_code[random_idx == 1]

cmap = plt.get_cmap('Set1').colors
patch_id = np.arange(feature_encode.shape[0])
plt.figure(figsize=(15, 8))
for i in range(5):
    plt.scatter(feature_encode[patch_name_code == i, 0], feature_encode[patch_name_code == i, 1], color=cmap[i],
                label=city_list[i], edgecolors='k')

for i in range(feature_encode.shape[0]):
    plt.text(feature_encode[i, 0], feature_encode[i, 1], patch_ids[i])
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('TSNE Projection Result')
plt.legend()
plt.tight_layout()
#plt.savefig(os.path.join(img_dir, 'res50_tsne_proj.png'))
plt.show()
