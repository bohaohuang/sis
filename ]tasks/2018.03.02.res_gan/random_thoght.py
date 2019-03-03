import os
import csv
import sis_utils
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

img_dir, task_dir = sis_utils.get_task_img_folder()
npy_file_name = os.path.join(task_dir, 'encoded_res50_2.npy')
city_order = {'aus': 0, 'chi': 1, 'kit': 2, 'tyr': 3, 'vie': 4}
city_list = ['austin', 'chicago', 'kitsap', 'tyrol-w', 'vienna']

file_name = os.path.join(task_dir, 'res50_fc1000_inria.csv')
features = []
with open(file_name, 'r') as f:
    csv_reader = csv.reader(f)
    cnt = 0
    for row in csv_reader:
        cnt += 1
        row = np.array(row)
        features.append(np.argsort(row)[::-1][:10])
        print(features[0])
        if cnt == 10:
            break

'''feature_encode = TSNE(n_components=2, perplexity=30, verbose=True).fit_transform(features)
np.save(npy_file_name, feature_encode)
feature_encode = np.array(feature_encode)

patch_name_fname = os.path.join(task_dir, 'res50_fc1000_inria.txt')
with open(patch_name_fname, 'r') as f:
    patch_name_list = f.readlines()
patch_name_list = [city_order[a[:3]] for a in patch_name_list]

patch_name_code = []
for i in patch_name_list:
    patch_name_code.append(i)
patch_name_code = np.array(patch_name_code)

cmap = plt.get_cmap('Set1').colors
plt.figure(figsize=(15, 8))
for i in range(5):
    plt.subplot(231+i)
    plt.scatter(feature_encode[patch_name_code == i, 0], feature_encode[patch_name_code == i, 1], color=cmap[i],
                label=city_list[i], edgecolors='k')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('TSNE Projection Result')
plt.legend()
#plt.savefig(os.path.join(img_dir, 'tsne_proj.png'))
plt.show()'''

