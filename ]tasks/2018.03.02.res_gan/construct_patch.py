import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from tqdm import tqdm
import utils


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
labels = KMeans(n_clusters=5, random_state=random_seed).fit_predict(features)
cmap = plt.get_cmap('Set1').colors

# load patch filelist
file_name = os.path.join(patchDir, 'fileList.txt')
with open(file_name, 'r') as f:
    files = f.readlines()

# construct patch file
patch_bucket = []
for i in range(5):
    patch_bucket.append([files[j] for j in range(len(files)) if labels[j] == i])
bucket = make_bucket_group(patch_bucket)
bucket_len = int(len(files)/5)

save_patch_flie_name = os.path.join(task_dir, 'fileList_{}.txt'.format(random_seed))
with open(save_patch_flie_name, 'w+') as f:
    for i in tqdm(range(bucket_len)):
        for j in range(len(bucket)):
            f.write(bucket[j][i])
