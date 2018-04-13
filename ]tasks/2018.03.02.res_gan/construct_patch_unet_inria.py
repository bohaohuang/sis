import os
import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from tqdm import tqdm
import utils
import uabCrossValMaker


def pick_most_different(dist, pick_num):
    idx_list = np.argsort(dist)
    idx_len = len(idx_list)
    pick_offset = min(idx_len-pick_num, int(idx_len*0.8))
    pick_range = idx_list[-pick_offset:]
    idx_random = np.random.permutation(len(pick_range))
    return idx_list[idx_random[:pick_num]].tolist()

def make_bucket_group(bucket):
    group_len = []
    # record length & random permute
    for cnt, b in enumerate(bucket):
        group_len.append(len(b))
        bucket[cnt] = [b[a] for a in np.random.permutation(group_len[-1])]
    offset = group_len - np.floor(np.mean(group_len))

    plus_term = []
    for cnt, b in enumerate(bucket):
        if offset[cnt] > 0:
            for i in range(int(offset[cnt])):
                plus_term.append(bucket[cnt].pop())
    plus_term = [plus_term[a] for a in np.random.permutation(len(plus_term))]
    for cnt, b in enumerate(bucket):
        if offset[cnt] < 0:
            for i in range(int(-offset[cnt])):
                bucket[cnt].append(plus_term.pop())
    b_len_min = int(np.min([len(a) for a in bucket]))
    return [b[:b_len_min] for b in bucket]


# settings
random_seed = 10
img_dir, task_dir = utils.get_task_img_folder()
file_name = os.path.join(task_dir, 'res50_fc1000_inria_unet.csv')
input_size = 572
patchDir = r'/hdd/uab_datasets/Results/PatchExtr/inria/chipExtrReg_cSz572x572_pad184'
npy_file_name = os.path.join(task_dir, 'encoded_res50_inria_unet.npy')
np.random.seed(random_seed)

# load tsne features
features = np.load(npy_file_name)
print(features.shape)
labels = KMeans(n_clusters=5, random_state=random_seed).fit_predict(features)
cmap = plt.get_cmap('Set1').colors
for i in range(5):
    plt.scatter(features[labels == i, 0], features[labels == i, 1], color=cmap[i], label=i, edgecolors='k')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('TSNE Projection Result')
plt.legend()
plt.tight_layout()
# plt.savefig(os.path.join(img_dir, 'kmenas_res50_tsne_proj.png'))
plt.show()

idx, file_list = uabCrossValMaker.uabUtilGetFolds(patchDir, 'fileList.txt', 'force_tile')
file_list_train = uabCrossValMaker.make_file_list_by_key(idx, file_list, [i for i in range(6, 37)])
file_list_train = [' '.join(a)+'\n' for a in file_list_train]
labels = [labels[i] for i in range(len(labels)) if idx[i] >= 6]

# construct patch file
print(len(file_list_train), len(labels))
patch_bucket = []
# city_list = ['austin', 'chicago', 'kitsap', 'tyrol-w', 'vienna']
cnt = np.zeros(5)
for i in range(5):
    patch_bucket.append([file_list_train[j] for j in range(len(file_list_train)) if labels[j] == i])
bucket = make_bucket_group(patch_bucket)
bucket_len = int(len(file_list_train)/5)

save_patch_flie_name = os.path.join(task_dir, 'unet_inria_fileList_{}.txt'.format(random_seed))
with open(save_patch_flie_name, 'w+') as f:
    for i in tqdm(range(bucket_len)):
        for j in range(len(bucket)):
            #replaced_line = re.sub(r'[a-z]{6,}(?=[\d]*_)', city_list[j], bucket[j][i])
            #cnt[j] += 1
            #f.write(replaced_line)
            f.write(bucket[j][i])
