import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def run_tsne(features, file_name, perplex=25, force_run=False):
    if not os.path.exists(file_name) or force_run:
        feature_encode = TSNE(n_components=2, perplexity=perplex, verbose=True).fit_transform(features)
        np.save(file_name, feature_encode)
    else:
        feature_encode = np.load(file_name)

    return feature_encode


def plot_tsne(feature_encode, patch_name_list, rand_percent=1):
    city_order = {'aus': 0, 'chi': 1, 'kit': 2, 'tyr': 3, 'vie': 4}
    city_list = ['austin', 'chicago', 'kitsap', 'tyrol-w', 'vienna']

    random_idx = np.random.binomial(1, rand_percent, feature_encode.shape[0])
    patch_name_list = [city_order[a[:3]] for a in patch_name_list]

    patch_name_code = []
    for i in patch_name_list:
        patch_name_code.append(i)
    patch_name_code = np.array(patch_name_code)
    patch_name_code = patch_name_code[random_idx == 1]

    cmap = plt.get_cmap('Set1').colors
    plt.figure(figsize=(15, 8))
    for i in range(5):
        plt.scatter(feature_encode[patch_name_code == i, 0], feature_encode[patch_name_code == i, 1], color=cmap[i],
                    label=city_list[i], edgecolors='k')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('TSNE Projection Result')
    plt.legend()
    # plt.axis('off')
    plt.tight_layout()
