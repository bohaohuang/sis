import os
import numpy as np
import matplotlib.pyplot as plt
import sis_utils
import ersa_utils


class StatsRecorder:
    def __init__(self, data=None):
        """
        data: ndarray, shape (nobservations, ndimensions)
        """
        if data is not None:
            data = np.atleast_2d(data)
            self.mean = data.mean(axis=0)
            self.std  = data.std(axis=0)
            self.nobservations = data.shape[0]
            self.ndimensions   = data.shape[1]
        else:
            self.nobservations = 0

    def update(self, data):
        """
        data: ndarray, shape (nobservations, ndimensions)
        """
        if self.nobservations == 0:
            self.__init__(data)
        else:
            data = np.atleast_2d(data)
            if data.shape[1] != self.ndimensions:
                raise ValueError("Data dims don't match prev observations.")

            newmean = data.mean(axis=0)
            newstd  = data.std(axis=0)

            m = self.nobservations * 1.0
            n = data.shape[0]

            tmp = self.mean

            self.mean = m/(m+n)*tmp + n/(m+n)*newmean
            self.std  = m/(m+n)*self.std**2 + n/(m+n)*newstd**2 +\
                        m*n/(m+n)**2 * (tmp - newmean)**2
            self.std  = np.sqrt(self.std)

            self.nobservations += n


class bayes_update:
    def __init__(self):
        self.m = 0
        self.mean = 0
        self.var = 0

    def update(self, d):
        n = d.shape[0]
        mu_n = np.mean(d)
        sig_n = np.var(d)
        factor_m = self.m / (self.m + n)
        factor_n = 1 - factor_m

        mean_update = factor_m * self.mean + factor_n * mu_n
        self.var = factor_m * (self.var + self.mean ** 2) + factor_n * (sig_n + mu_n ** 2) - mean_update ** 2
        self.mean = mean_update

        self.m += n

        return np.array([self.mean, self.var])


def get_shift_vals(act_dict_train, act_dict_valid):
    shift_dict = dict()
    layer_mean_train = [[] for _ in range(19)]
    layer_mean_valid = [[] for _ in range(19)]

    for act_name, up_train in act_dict_train.items():
        up_valid = act_dict_valid[act_name]
        layer_id = int(act_name.split('_')[1])
        layer_mean_train[layer_id].append(up_train.mean)
        layer_mean_valid[layer_id].append(up_valid.mean)
    layer_mean_train = [np.mean(layer_mean_train[i]) for i in range(19)]
    layer_mean_valid = [np.mean(layer_mean_valid[i]) for i in range(19)]

    for act_name, up_train in act_dict_train.items():
        layer_id = int(act_name.split('_')[1])
        up_valid = act_dict_valid[act_name]
        scale = np.sqrt(up_train.var / up_valid.var)
        shift_1 = layer_mean_valid[layer_id]
        shift_2 = layer_mean_train[layer_id]
        shift_dict[act_name] = np.array([scale, shift_1, shift_2])
        print(shift_dict[act_name])
    return shift_dict


def get_shift_vals2(act_dict_train, act_dict_valid):
    """
    Compute channel-wise mean and std here
    :param act_dict_train:
    :param act_dict_valid:
    :return:
    """
    shift_dict = dict()
    for act_name, up_train in act_dict_train.items():
        up_valid = act_dict_valid[act_name]
        scale = np.sqrt(up_train.var / up_valid.var)
        shift_1 = up_valid.mean
        shift_2 = up_train.mean
        shift_dict[act_name] = np.array([scale, shift_1, shift_2])
        print(shift_dict[act_name])
    return shift_dict


def get_shift_vals3(act_dict_train, act_dict_valid):
    """
    Compute channel-wise mean and std here
    :param act_dict_train:
    :param act_dict_valid:
    :return:
    """
    shift_dict = dict()
    for act_name, up_train in act_dict_train.items():
        up_valid = act_dict_valid[act_name]
        scale = up_train.std[0] / up_valid.std[0]
        shift_1 = up_valid.mean[0]
        shift_2 = up_train.mean[0]
        shift_dict[act_name] = np.array([scale, shift_1, shift_2])
        print(shift_dict[act_name])
    return shift_dict


if __name__ == '__main__':
    plt.figure(figsize=(10, 12))

    img_dir, task_dir = sis_utils.get_task_img_folder()
    city_name = 'DC'

    path_to_save = os.path.join(task_dir, 'dtda_new', city_name, 'valid')
    save_name = os.path.join(path_to_save, 'activation_list.pkl')

    act_dict_valid = ersa_utils.load_file(save_name)
    m_list = []
    v_list = []
    for act_name, up in act_dict_valid.items():
        m_list.append(up.mean)
        v_list.append(up.var)

    ax1 = plt.subplot(411)
    plt.plot(m_list, label='valid')
    ax2 = plt.subplot(413, sharex=ax1)
    plt.plot(v_list, label='valid')

    path_to_save = os.path.join(task_dir, 'dtda_new', city_name, 'train')
    save_name = os.path.join(path_to_save, 'activation_list.pkl')

    act_dict_train = ersa_utils.load_file(save_name)
    m_list = []
    v_list = []
    for act_name, up in act_dict_train.items():
        m_list.append(up.mean)
        v_list.append(up.var)

    ax3 = plt.subplot(412, sharex=ax1, sharey=ax1)
    plt.plot(m_list, label='train')
    ax4 = plt.subplot(414, sharex=ax2, sharey=ax2)
    plt.plot(v_list, label='train')

    plt.tight_layout()
    plt.show()

    shift_dict = get_shift_vals(act_dict_train, act_dict_valid)
    path_to_save = os.path.join(task_dir, 'dtda_new', city_name, 'shift_dict.pkl')
    ersa_utils.save_file(path_to_save, shift_dict)
