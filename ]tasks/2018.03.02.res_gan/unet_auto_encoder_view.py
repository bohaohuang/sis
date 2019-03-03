import os
import csv
import pickle
import imageio
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import sis_utils

run_clustering = False
npy_file_name = os.path.join(r'/hdd6/temp', 'encoded_uencoder.npy')
cmap = plt.get_cmap('Set1').colors
city_order = ['austin', 'chicago', 'kitsap', 'tyrol-w', 'vienna']
patch_dir = r'/hdd/uab_datasets/Results/PatchExtr/inria/chipExtrReg_cSz572x572_pad184'
input_size = 572
img_dir, task_dir = sis_utils.get_task_img_folder()

if run_clustering:
    file_name = os.path.join(r'/hdd6/temp', 'encoded_uencoder.csv')
    features = []
    with open(file_name, 'r') as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            features.append(row)

    feature_encode = TSNE(n_components=2, perplexity=20).fit_transform(features)
    np.save(npy_file_name, feature_encode)
else:
    feature_encode = np.load(npy_file_name)

city_num = 5
sample_num = feature_encode.shape[0]
file_name = os.path.join(r'/hdd6/temp', 'encoded_uencoder_city_list.pkl')
with open(file_name, 'rb') as handle:
    city_list = pickle.load(handle)

fig = plt.figure(figsize=(15, 8))
for i in range(city_num):
    point_idx = np.arange(i, sample_num, step=city_num)
    plt.scatter(feature_encode[point_idx, 0], feature_encode[point_idx, 1], color=cmap[i],
                label=city_order[i], edgecolors='k')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('TSNE Projection Result')
plt.legend()
plt.savefig(os.path.join(img_dir, 'tsne_proj.png'))
#plt.show()
plt.close(fig)

view_id = [1982, 482,
           989, 2664, 3579, 4389, 3489,
           2268, 3133, 1198,
           4371, 3596,
           4335, 4240, 2555]
exts = ['RGB0.jpg', 'RGB1.jpg', 'RGB2.jpg']
for vid in view_id:
    img = np.zeros((input_size, input_size, 3), dtype=np.uint8)
    for cnt, ext in enumerate(exts):
        file_name = city_list[vid][:-8] + ext
        img[:, :, cnt] = imageio.imread(os.path.join(patch_dir, file_name))
    imageio.imsave(os.path.join(img_dir, '{}_patch.png'.format(vid)), img)
