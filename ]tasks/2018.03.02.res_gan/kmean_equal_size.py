import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import utils


def show_cluster(data, labels):
    cmap = plt.get_cmap('Set1').colors
    for i in np.unique(labels):
        plt.scatter(data[labels == i, 0], data[labels == i, 1], color=cmap[i], label=i, edgecolors='k')
    plt.show()


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
            means   [i, :] = np.mean(data[labels==i, :], axis=0)

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


if __name__ == '__main__':
    img_dir, task_dir = utils.get_task_img_folder()
    npy_file_name = os.path.join(task_dir, 'encoded_res50_2.npy')
    feature_encode = np.load(npy_file_name)
    feature_encode = np.array(feature_encode)

    '''for i in range(5):
        plt.scatter(feature_encode[:, 0], feature_encode[:, 1],  edgecolors='k')
    plt.show()'''

    kequal = kmeans_equal(5)
    kequal.fit(feature_encode)
