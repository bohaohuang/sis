import os
import time
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import utils


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

    def fit(self, step_num=10, th=1e-5):
        # init points
        centers = np.zeros((self.cluster_num, self.feature_dim))
        labels = KMeans(n_clusters=5, random_state=random_seed).fit_predict(features)
        plt.hist(labels)
        plt.show()
        for i in range(self.cluster_num):
            centers[i, :] = np.mean(self.data[labels == i, :], axis=0)

        # start fitting
        cnt = 0
        eps = np.inf
        while cnt < step_num and eps > th:
            start_time = time.time()

            labels = -1 * np.ones(self.n_sample)
            for i in range(self.cluster_num):
                dist = np.sum((self.data - centers[i, :]) ** 2, axis=1)
                not_i = [c for c in range(self.cluster_num) if c != i]
                for c in not_i:
                    dist -= np.sum((self.data - centers[c, :]) ** 2, axis=1)
                close_idx = np.argsort(dist)
                group_cnt = 0
                j = 0
                while group_cnt < self.group_len:
                    if labels[close_idx[j]] == -1:
                        labels[close_idx[j]] = i
                        group_cnt +=1
                        j += 1
                    else:
                        j += 1

            # update centers
            for i in range(self.cluster_num):
                centers[i, :] = np.mean(self.data[labels == i, :], axis=0)
            centers = np.array([centers[i] for i in np.random.permutation(self.cluster_num)])
            cnt += 1

            for i in range(self.cluster_num):
                plt.scatter(self.data[labels == i, 0], self.data[labels == i, 1])
            plt.show()

            print('step {} complete, duration={:.2f}'.format(cnt, time.time()-start_time))


if __name__ == '__main__':
    # read data
    random_seed = 9
    img_dir, task_dir = utils.get_task_img_folder()
    file_name = os.path.join(task_dir, 'res50_fc1000_inria.csv')
    input_size = 321
    patchDir = r'/hdd/uab_datasets/Results/PatchExtr/inria/chipExtrReg_cSz321x321_pad0'
    npy_file_name = os.path.join(task_dir, 'encoded_res50_2.npy')
    np.random.seed(random_seed)

    # load tsne features
    features = np.load(npy_file_name)
    esc = equal_size_clustering(features, 5)
    esc.fit_angle()

