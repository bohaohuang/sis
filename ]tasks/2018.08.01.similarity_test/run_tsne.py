import os
import imageio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from mpl_toolkits.axes_grid1 import Grid
from make_res50_features import crop_center


def run_tsne(features, file_name, perplex=25, force_run=False, learn_rate=200):
    if not os.path.exists(file_name) or force_run:
        feature_encode = TSNE(n_components=2, perplexity=perplex, learning_rate=learn_rate, verbose=True).\
            fit_transform(features)
        np.save(file_name, feature_encode)
    else:
        feature_encode = np.load(file_name)

    return feature_encode


def plot_tsne(feature_encode, patch_name_list, rand_percent=1, show_id=False):
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

    if show_id:
        patch_ids = np.arange(feature_encode.shape[0])
        patch_ids = patch_ids[random_idx == 1]
        for i in range(feature_encode.shape[0]):
            plt.text(feature_encode[i, 0], feature_encode[i, 1], patch_ids[i])

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('TSNE Projection Result')
    plt.legend()
    plt.tight_layout()


def view_encoded_patches(img_ids, files, patchDir, model_name, img_dir, cust_name=''):
    if model_name == 'unet':
        input_size = 572
    else:
        input_size = 321
    fig = plt.figure(figsize=(11, 2))
    grid = Grid(fig, rect=111, nrows_ncols=(1, 6), axes_pad=0.1, label_mode='L')
    for plt_cnt, iid in enumerate(img_ids):
        img = np.zeros((input_size, input_size, 3), dtype=np.uint8)
        for cnt in range(3):
            img[:, :, cnt] = imageio.imread(os.path.join(patchDir, files[iid][:-1]+'_RGB{}.jpg'.format(cnt)))
        grid[plt_cnt].imshow(crop_center(img, 224, 224))
        grid[plt_cnt].set_axis_off()
    plt.tight_layout()
    if not os.path.exists(os.path.join(img_dir, cust_name)):
        os.makedirs(os.path.join(img_dir, cust_name))
    plt.savefig(os.path.join(img_dir, cust_name, '{}_{}.png'.format(model_name, '_'.join([str(a) for a in img_ids]))))
    plt.show()
