import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import Grid
import utils
from make_res50_features import make_res50_features
from run_tsne import run_tsne, plot_tsne
from city_building_truth import make_building_truth, make_city_truth
from gmm_cluster import *

model_name = 'deeplab'
perplex = 25
do_tsne = False
do_bic = True
show_bic = False
city_select = [0, 1, 4]

# 1. make features
img_dir, task_dir = utils.get_task_img_folder()
feature_file_name, patch_file_name, ps, patchDir, idx = make_res50_features(model_name, task_dir, GPU=0,
                                                                            force_run=False)
feature = pd.read_csv(feature_file_name, sep=',', header=None).values
with open(patch_file_name, 'r') as f:
    patch_names = f.readlines()

if do_tsne:
    # (2. do tsne)
    file_name = os.path.join(task_dir, '{}_inria_p{}.npy'.format(model_name, perplex))
    feature_encode = run_tsne(feature, file_name, perplex=perplex, force_run=False)
    plot_tsne(feature_encode, patch_names, rand_percent=1)

# 2. make city and building truth
truth_city = make_city_truth(task_dir, model_name, patch_names, force_run=False)
truth_building = make_building_truth(ps, task_dir, model_name, patchDir, patch_names, force_run=False)

if do_bic:
    # 3. do bic
    bic, test_points = gmm_bic_test(idx, feature, task_dir, truth_building, file_name='llh_bic_test.npy',
                                    test_comp=None, force_run=False)
    n_comp = test_points[np.argmin(bic)]
    if show_bic:
        plt.plot(test_points, bic)
        plt.show()
else:
    n_comp = 40

# 3. train GMM model
city_list = ['Aus', 'Chi', 'Kit', 'Tyr', 'Vie']
fig = plt.figure(figsize=(12, 8))
grid = Grid(fig, rect=111, nrows_ncols=(5, 5), axes_pad=0.25, label_mode='L', share_all=True)
ax_cnt = 0
for city_1 in range(5):
    for city_2 in range(5):
        city_select = [i for i in range(5) if i != city_1 and i != city_2]
        gmm = train_gmm_source_domain(task_dir, idx, feature, truth_city, truth_building, city_select, n_comp,
                                      force_run=False)
        llh = test_gmm_model(idx, patch_names, gmm, feature)
        str2show = 'No {}'.format(city_list[city_2])
        ylab2show = 'No {}'.format(city_list[city_1])
        if city_1 == 0:
            plot_llh(llh, city_select, ax=grid[ax_cnt], title=str2show, ylab=ylab2show)
        else:
            plot_llh(llh, city_select, ax=grid[ax_cnt], ylab=ylab2show)
        ax_cnt += 1
plt.tight_layout()
plt.show()
