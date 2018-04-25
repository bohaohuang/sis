import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from tqdm import tqdm
import utils
import uabCrossValMaker


class kmeans_equal(object):
    def __init__(self, n_cluster, ransed=1004):
        self.n_cluster = n_cluster
        self.ransed = ransed
        np.random.seed(self.ransed)

    def fit(self, data, step_num=50):
        # 1. Compute the desired cluster size, n/k.
        n_sample = data.shape[0]
        group_size = n_sample // self.n_cluster
        group_cnt = np.zeros(self.n_cluster)

        # 2. Initialize means, preferably with k-means++
        labels = KMeans(n_clusters=5, random_state=self.ransed).fit_predict(data)
        means = np.zeros((self.n_cluster, data.shape[1]))
        for i in range(self.n_cluster):
            means[i, :] = np.mean(data[labels==i, :], axis=0)

        # 3. Order points by the distance to their nearest cluster minus distance to the farthest cluster
        point_dist = np.zeros((n_sample, self.n_cluster))
        sort_dist = np.zeros(n_sample)
        for i in range(n_sample):
            point_dist[i, :] = np.sum((means - data[i, :]) ** 2, axis=1)
            sort_dist[i] = np.min(point_dist[i, :]) - np.max(point_dist[i, :])
        sort_idx = np.argsort(sort_dist)

        # 4. Assign points to their preferred cluster until this cluster is full, then resort remaining objects,
        # without taking the full cluster into account anymore
        for i in sort_idx:
            cluster_dist_idx = np.argsort(point_dist[i, :])
            for j in range(self.n_cluster):
                if group_cnt[cluster_dist_idx[j]] < group_size:
                    labels[i] = cluster_dist_idx[j]
                    group_cnt[cluster_dist_idx[j]] += 1
                    break

        # iterations
        for step_cnt in range(step_num):
            # 1. Compute current cluster means
            means = np.zeros((self.n_cluster, data.shape[1]))
            for i in range(self.n_cluster):
                means[i, :] = np.mean(data[labels == i, :], axis=0)

            # 2. For each object, compute the distances to the cluster means
            point_dist = np.zeros((n_sample, self.n_cluster))
            sort_dist = np.zeros(n_sample)
            for i in range(n_sample):
                point_dist[i, :] = np.sum((means - data[i, :]) ** 2, axis=1)
                sort_dist[i] = np.min(point_dist[i, :]) - point_dist[i, labels[i]]

            # 3. Sort elements based on the delta of the current assignment and the best possible alternate assignment.
            sort_idx = np.argsort(sort_dist)

            # 4. For each element by priority
            swap_cnt = 0
            outgoing_list = [[] for i in range(self.n_cluster)]
            for i in sort_idx:
                if np.argsort(point_dist[i, :])[0] != labels[i]:
                    # indicate a swap proposal for the other cluster
                    outgoing_list[np.argsort(point_dist[i, :])[0]].append(i)
                    if outgoing_list[labels[i]]:
                        # If there is a swap proposal from the other cluster (or any cluster with a lower distance),
                        # swap the two element cluster assignments
                        dist_swap = []
                        for x in outgoing_list[labels[i]]:
                            new_dist = point_dist[i, np.argsort(point_dist[i, :])[0]] + point_dist[x, labels[i]]
                            old_dist = point_dist[i, labels[i]] + point_dist[x, labels[x]]
                            assert (new_dist-old_dist<0)
                            dist_swap.append(new_dist - old_dist)
                        point2swap = np.argsort(dist_swap)[0]
                        temp = labels[i]
                        labels[i] = labels[outgoing_list[labels[i]][point2swap]]
                        labels[outgoing_list[temp][point2swap]] = temp
                        outgoing_list[np.argsort(point_dist[i, :])[0]].remove(i)
                        outgoing_list[temp].remove(outgoing_list[temp][point2swap])
                        swap_cnt += 1
            if swap_cnt == 0:
                print('Converges at step {}'.format(step_cnt))
                break
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
for random_seed in range(5):
    img_dir, task_dir = utils.get_task_img_folder()
    file_name = os.path.join(task_dir, 'res50_fc1000_inria.csv')
    input_size = 321
    patchDir = r'/hdd/uab_datasets/Results/PatchExtr/inria/chipExtrReg_cSz321x321_pad0'
    npy_file_name = os.path.join(task_dir, 'encoded_res50_2.npy')
    np.random.seed(random_seed)
    # load tsne features
    features = []
    with open(file_name, 'r') as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            features.append(row)
    features = np.array(features).astype(np.float32)
    labels = kmeans_equal(5).fit(features, step_num=500)

    features = np.load(npy_file_name)
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
    labels = [labels[i] for i in range(labels.shape[0]) if idx[i] >= 6]

    # construct patch file
    patch_bucket = []
    city_list = ['austin', 'chicago', 'kitsap', 'tyrol-w', 'vienna']
    cnt = np.zeros(5)
    for i in range(5):
        patch_bucket.append([file_list_train[j] for j in range(len(file_list_train)) if labels[j] == i])
    bucket = make_bucket_group(patch_bucket)
    bucket_len = int(len(file_list_train)/5)

    save_patch_flie_name = os.path.join(task_dir, 'deeplab_inria_cp1000_{}.txt'.format(random_seed))
    with open(save_patch_flie_name, 'w+') as f:
        for i in tqdm(range(bucket_len)):
            for j in range(len(bucket)):
                #replaced_line = re.sub(r'[a-z]{6,}(?=[\d]*_)', city_list[j], bucket[j][i])
                #cnt[j] += 1
                #f.write(replaced_line)
                f.write(bucket[j][i])
