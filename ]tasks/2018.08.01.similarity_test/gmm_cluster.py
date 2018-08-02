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


def train_gmm_source_domain(task_dir, idx, feature, truth_city, truth_building, city_select, n_comp, force_run=False):
    file_name = os.path.join(task_dir, 'gmm_n{}_{}.pkl'.format(n_comp, '_'.join(str(a) for a in city_select)))
    if not os.path.exists(file_name) or force_run:
        idx = np.array(idx)
        truth_city_train = truth_city[idx >= 6]
        feature_train = feature[idx >= 6, :]

        train_idx = []
        for s in city_select:
            train_idx.append(truth_city_train == s)
        train_idx = np.any(train_idx, axis=0)
        train_idx = np.all([train_idx, truth_building[idx >= 6] == 1], axis=0)
        gmm_model = GaussianMixture(n_components=n_comp, covariance_type='diag')
        gmm_model.fit(feature_train[train_idx, :])
        with open(file_name, 'wb') as f:
            pickle.dump(gmm_model, f)
    else:
        with open(file_name, 'rb') as f:
            gmm_model = pickle.load(f)

    return gmm_model


def test_gmm_model(idx, patch_names, gmm, feature):
    city_dict = {'aus': 0, 'chi': 1, 'kit': 2, 'tyr': 3, 'vie': 4}
    feature = feature[np.array(idx) < 6, :]
    patch_valid = [patch_names[i] for i in range(len(patch_names)) if idx[i] < 6]
    city_name_list = [a[:3] for a in patch_valid]
    city_id_list = [city_dict[a] for a in city_name_list]
    llh = np.zeros(5)
    for test_city in tqdm(range(5)):
        test_city_feature = feature[[i for i in range(len(city_id_list)) if city_id_list[i] == test_city], :]
        llh[test_city] = gmm.score(test_city_feature)
    return llh


def plot_llh(llh, train_idx, ax=None, title=False, ylab=None):
    if ax is None:
        ax = plt.subplot(111)

    city_list = ['austin', 'chicago', 'kitsap', 'tyrol-w', 'vienna']
    normalized_data = softmax(llh, t=1000)
    ax.bar(np.arange(5), normalized_data)
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
        ax.text(cnt - 0.25, l, '{:.3f}'.format(l), fontsize=8)


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
