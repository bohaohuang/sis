import os
import csv
import imageio
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.manifold import TSNE
import utils

run_clustering = True
img_dir, task_dir = utils.get_task_img_folder()
npy_file_name = os.path.join(task_dir, 'encoded_res50_inria_unet_crop.npy')
city_order = {'aus': 0, 'chi': 1, 'kit': 2, 'tyr': 3, 'vie': 4}
city_list = ['austin', 'chicago', 'kitsap', 'tyrol-w', 'vienna']
np.random.seed(1004)

if run_clustering:
    file_name = os.path.join(task_dir, 'temp', 'res50_fc1000_inria_unet_crop.csv')
    features = []
    with open(file_name, 'r') as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            features.append(row)
    print(len(features))
    feature_encode = TSNE(n_components=2, perplexity=25, verbose=True).fit_transform(features)
    np.save(npy_file_name, feature_encode)
else:
    feature_encode = np.load(npy_file_name)
feature_encode = np.array(feature_encode)

random_idx = np.random.binomial(1, 1, feature_encode.shape[0])
patch_ids = np.arange(feature_encode.shape[0])
feature_encode = feature_encode[random_idx == 1, :]
patch_ids = patch_ids[random_idx == 1]

patch_name_fname = os.path.join(task_dir, 'temp', 'res50_fc1000_inria_unet_crop.txt')
with open(patch_name_fname, 'r') as f:
    patch_name_list = f.readlines()

patch_percent_list = [patch_name_list[i] for i in range(len(patch_name_list)) if random_idx[i] == 1]
patch_percent = np.zeros(len(patch_percent_list))
patchDir = r'/hdd/uab_datasets/Results/PatchExtr/inria/chipExtrReg_cSz572x572_pad92'
for cnt, patch_name in enumerate(tqdm(patch_percent_list)):
    gt = imageio.imread(os.path.join(patchDir, '{}_GT_Divide.png'.format(patch_name.strip())))
    patch_percent[cnt] = np.sum(gt)/(gt.shape[0] * gt.shape[1])

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
#plt.scatter(feature_encode[:, 0], feature_encode[:, 1], c=patch_percent, cmap=plt.get_cmap('bwr'))
#plt.colorbar()

for i in range(feature_encode.shape[0]):
    plt.text(feature_encode[i, 0], feature_encode[i, 1], patch_ids[i])
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('TSNE Projection Result')
plt.legend()
plt.tight_layout()
#plt.savefig(os.path.join(img_dir, 'res50_tsne_proj_unet_inria.png'))
plt.show()
