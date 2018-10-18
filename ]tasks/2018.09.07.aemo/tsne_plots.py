import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import utils
import ersa_utils
import processBlock


def get_tsne_features(perplex=40, learn_rate=200):
    feature_encode = TSNE(n_components=2, perplexity=perplex, learning_rate=learn_rate, verbose=True).\
        fit_transform(features)
    return feature_encode


# settings
img_dir, task_dir = utils.get_task_img_folder()
show_figure = False
perplex = 40
learn_rate = 200

aemo_img_dir = os.path.join(img_dir, 'aemo_hist_patches')
aemo_ftr_dir = os.path.join(task_dir, 'aemo_hist_patches')

spca_img_dir = r'/hdd/ersa/patch_extractor/spca_all'
spca_ftr_dir = os.path.join(task_dir, 'spca_patches')

# load aemo features
aemo_patch_name_file = os.path.join(aemo_ftr_dir, 'res50_patches.txt')
aemo_patch_names = ersa_utils.load_file(aemo_patch_name_file)

aemo_feature_name_file = os.path.join(aemo_ftr_dir, 'res50_feature.csv')
aemo_feature = np.genfromtxt(aemo_feature_name_file, delimiter=',')

# load spca features
spca_patch_name_file = os.path.join(spca_ftr_dir, 'res50_patches.txt')
spca_patch_names = ersa_utils.load_file(spca_patch_name_file)

spca_feature_name_file = os.path.join(spca_ftr_dir, 'res50_feature.csv')
spca_feature = np.genfromtxt(spca_feature_name_file, delimiter=',')

# do tsne
features = np.concatenate([aemo_feature, spca_feature], axis=0)
save_file = os.path.join(task_dir, 'hist_tsne_pp{}_lr{}.npy'.format(perplex, learn_rate))
feature_encode = processBlock.ValueComputeProcess('hist_tsnp_pp{}_lr{}'.format(perplex, learn_rate), task_dir, save_file,
                                                  lambda: get_tsne_features(perplex, learn_rate)).run().val

aemo_num = len(aemo_patch_names)

# show figure
plt.figure(figsize=(8, 6))
plt.scatter(feature_encode[:aemo_num, 0], feature_encode[:aemo_num, 1], label='AEMO')
plt.scatter(feature_encode[aemo_num:, 0], feature_encode[aemo_num:, 1], label='California')
plt.xlabel('Feature 0')
plt.ylabel('Feature 1')
plt.legend()
plt.title('TSNE Projection with ResNet50 Features')
plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'hist_tsne_project_pp{}_lr{}.png'.format(perplex, learn_rate)))
plt.show()
