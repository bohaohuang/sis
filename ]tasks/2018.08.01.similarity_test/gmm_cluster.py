import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.mixture import GaussianMixture


def softmax(x, t=1):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x/t - np.max(x/t))
    return e_x / e_x.sum(axis=0)


def gmm_bic_test(idx, feature, task_dir, truth_building, file_name='llh_bic_test.npy', test_comp=None, force_run=False):
    save_file_name = os.path.join(task_dir, file_name)
    if test_comp is None:
        test_comp = list(range(10, 151, 10)) + list(range(160, 501, 20))

    if not os.path.exists(save_file_name) or force_run:

        idx = np.array(idx)
        feature_train = feature[idx >= 6, :]

        bic = np.zeros(len(test_comp))
        pbar = tqdm(test_comp)
        for cnt, n_comp in enumerate(pbar):
            gmm_model = GaussianMixture(n_components=n_comp, covariance_type='diag')
            gmm_model.fit(feature_train[truth_building[idx >= 6] == 1, :])
            bic[cnt] = gmm_model.bic(feature[idx < 6, :])
        np.save(save_file_name, bic)
    else:
        bic = np.load(save_file_name)
    return bic, test_comp


def select_feature(feature, train_select, truth_city, truth_building, city_select, only_building):
    truth_city_train = truth_city[train_select]
    feature_train = feature[train_select, :]

    train_idx = []
    for s in city_select:
        train_idx.append(truth_city_train == s)
    train_idx = np.any(train_idx, axis=0)
    if only_building:
        train_idx = np.all([train_idx, truth_building[train_select] == 1], axis=0)

    return feature_train[train_idx, :]


def train_gmm(task_dir, train_select, feature, truth_city, truth_building, city_select, n_comp, force_run=False,
              only_building=True):
    if only_building:
        file_name = os.path.join(task_dir, 's{}_gmm_n{}_{}.pkl'.format(sum(train_select), n_comp,
                                                                       '_'.join(str(a) for a in city_select)))
    else:
        file_name = os.path.join(task_dir, 's{}_gmm_n{}_{}_all.pkl'.format(sum(train_select), n_comp,
                                                                           '_'.join(str(a) for a in city_select)))
    if not os.path.exists(file_name) or force_run:
        feature_train = select_feature(feature, train_select, truth_city, truth_building, city_select, only_building)
        gmm_model = GaussianMixture(n_components=n_comp, covariance_type='diag')
        gmm_model.fit(feature_train)
        with open(file_name, 'wb') as f:
            pickle.dump(gmm_model, f)
    else:
        with open(file_name, 'rb') as f:
            gmm_model = pickle.load(f)

    return gmm_model


def train_gmm_source_domain(task_dir, idx, feature, truth_city, truth_building, city_select, n_comp, force_run=False,
                            only_building=True):
    if only_building:
        file_name = os.path.join(task_dir, 'gmm_n{}_{}.pkl'.format(n_comp, '_'.join(str(a) for a in city_select)))
    else:
        file_name = os.path.join(task_dir, 'gmm_n{}_{}_all.pkl'.format(n_comp, '_'.join(str(a) for a in city_select)))
    if not os.path.exists(file_name) or force_run:
        idx = np.array(idx)
        truth_city_train = truth_city[idx >= 6]
        feature_train = feature[idx >= 6, :]

        train_idx = []
        for s in city_select:
            train_idx.append(truth_city_train == s)
        train_idx = np.any(train_idx, axis=0)
        if only_building:
            train_idx = np.all([train_idx, truth_building[idx >= 6] == 1], axis=0)
        gmm_model = GaussianMixture(n_components=n_comp, covariance_type='diag')
        gmm_model.fit(feature_train[train_idx, :])
        with open(file_name, 'wb') as f:
            pickle.dump(gmm_model, f)
    else:
        with open(file_name, 'rb') as f:
            gmm_model = pickle.load(f)

    return gmm_model


def test_gmm_model(idx, patch_names, gmm, feature, test_select=None, use_bic=False, test_city=range(5)):
    if test_select is None:
        test_select = np.array(idx) < 6
    city_dict = {'aus': 0, 'chi': 1, 'kit': 2, 'tyr': 3, 'vie': 4}
    feature = feature[test_select, :]
    patch_valid = [patch_names[i] for i in range(len(patch_names)) if test_select[i]]
    city_name_list = [a[:3] for a in patch_valid]
    city_id_list = [city_dict[a] for a in city_name_list]
    llh = np.zeros(len(test_city))
    bic = 0
    for cnt, test_city in enumerate(tqdm(test_city)):
        test_city_feature = feature[[i for i in range(len(city_id_list)) if city_id_list[i] == test_city], :]
        llh[cnt] = gmm.score(test_city_feature)
        if use_bic:
            bic += gmm.score(test_city_feature)
    if use_bic:
        return llh, bic
    else:
        return llh


def test_gmm_model_sample_wise(idx, gmm, feature, truth_city, city_select, test_select=None):
    if test_select is None:
        test_select = np.array(idx) < 6
    train_idx = []
    for s in city_select:
        train_idx.append(truth_city == s)
    train_idx = np.any(train_idx, axis=0)
    train_idx = np.all([train_idx, test_select], axis=0)
    return gmm.score_samples(feature[train_idx, :]), train_idx


def plot_sample_wise_llh(llh, train_idx, truth_city, t=1000, city_range=range(5)):
    city_list = ['austin', 'chicago', 'kitsap', 'tyrol-w', 'vienna']
    llh_norm = softmax(llh, t)
    sort_idx = np.argsort(llh_norm)[::-1]
    truth_city = truth_city[train_idx]
    truth_city = truth_city[sort_idx]
    for city_cnt in city_range:
        city_ind = np.arange(llh_norm.shape[0])[truth_city == city_cnt]
        plt.bar(city_ind, llh_norm[sort_idx[city_ind]], label=city_list[city_cnt])
    plt.legend()
    plt.tight_layout()


def plot_llh(llh, train_idx, ax=None, title=False, ylab=None, t=1000, x_pos=np.arange(5)):
    if ax is None:
        ax = plt.subplot(111)

    city_list = ['austin', 'chicago', 'kitsap', 'tyrol-w', 'vienna']
    normalized_data = softmax(llh, t=t)
    ax.bar(x_pos, normalized_data)
    title_str = ' '.join([city_list[a] for a in train_idx])
    if title is False:
        pass
    elif title is None:
        ax.set_title('Train on [{}]'.format(title_str))
    else:
        ax.set_title(title)
    if ylab is not None:
        ax.set_ylabel(ylab)
    ax.set_xticks(np.arange(5))
    ax.set_xticklabels(['aus', 'chi', 'kit', 'tyr', 'vie'])
    for cnt, l in enumerate(normalized_data):
        ax.text(x_pos[cnt] - 0.25, l, '{:.3f}'.format(l), fontsize=8)


def test_correlation(task_dir, idx, feature, truth_city, truth_building, n_comp, patch_names, force_run=False):
    corr_matrix = np.zeros((5, 5, 5))
    for city_1 in range(5):
        for city_2 in range(city_1, 5):
            city_select = [a for a in range(5) if a != city_1 and a != city_2]
            gmm = train_gmm_source_domain(task_dir, idx, feature, truth_city, truth_building, city_select, n_comp,
                                          force_run=force_run)
            llh = test_gmm_model(idx, patch_names, gmm, feature)
            for test_city in range(5):
                corr_matrix[test_city, city_1, city_2] = llh[test_city]
    return corr_matrix
