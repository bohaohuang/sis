import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from tqdm import tqdm
import utils
import uabCrossValMaker


class equal_size_clustering(object):
    def __init__(self, data, cluster_num):
        self.data = data
        self.cluster_num = cluster_num
        self.n_sample = data.shape[0]
        self.feature_dim = data.shape[1]
        self.group_len = int(np.floor(self.n_sample/self.cluster_num))

    def fit_angle(self, seed=1004):
        np.random.seed(seed)
        # get angles of each point
        angles = np.zeros(self.n_sample)
        for i in range(self.n_sample):
            pt = self.data[i, :]
            if pt[0] >= 0 and pt[1] >= 0:
                angles[i] = np.arctan(pt[1] / pt[0])
            elif pt[0] < 0 and pt[1] >= 0:
                angles[i] = np.pi - np.arctan(-pt[1] / pt[0])
            elif pt[0] < 0 and pt[1] < 0:
                angles[i] = np.arctan(-pt[1] / -pt[0]) + np.pi
            else:
                angles[i] = 2*np.pi - np.arctan(pt[1] / -pt[0])

        offset = np.random.uniform(0, np.pi/2)
        angles = angles - offset
        angles[angles < 0] = np.pi*2 - offset

        ang_idx = np.argsort(angles)
        labels = np.zeros(self.n_sample)
        for i in range(self.cluster_num):
            labels[ang_idx[self.group_len*i:self.group_len*(i+1)]] = i

        return labels


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
    offset = group_len - np.mean(group_len)

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
    return bucket


# settings
random_seed = 4
img_dir, task_dir = utils.get_task_img_folder()
file_name = os.path.join(task_dir, 'res50_fc1000_inria.csv')
input_size = 321
patchDir = r'/hdd/uab_datasets/Results/PatchExtr/inria/chipExtrReg_cSz321x321_pad0'
npy_file_name = os.path.join(task_dir, 'encoded_res50_2.npy')
np.random.seed(random_seed)

# load tsne features
features = np.load(npy_file_name)
labels = equal_size_clustering(data=features, cluster_num=5).fit_angle(seed=random_seed)
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
patch_bucket = []
city_list = ['austin', 'chicago', 'kitsap', 'tyrol-w', 'vienna']
cnt = np.zeros(5)
for i in range(5):
    patch_bucket.append([file_list_train[j] for j in range(len(file_list_train)) if labels[j] == i])
bucket = make_bucket_group(patch_bucket)
bucket_len = int(len(file_list_train)/5)

save_patch_flie_name = os.path.join(task_dir, 'deeplab_inria_cp_{}.txt'.format(random_seed))
with open(save_patch_flie_name, 'w+') as f:
    for i in tqdm(range(bucket_len)):
        for j in range(len(bucket)):
            #replaced_line = re.sub(r'[a-z]{6,}(?=[\d]*_)', city_list[j], bucket[j][i])
            #cnt[j] += 1
            #f.write(replaced_line)
            f.write(bucket[j][i])
