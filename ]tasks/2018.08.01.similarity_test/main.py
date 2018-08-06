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
only_building = True

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
    plt.savefig(os.path.join(img_dir, 'tsne_unet_inria_n{}_all'.format(perplex)))
    plt.show()

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
'''fig = plt.figure(figsize=(12, 8))
grid = Grid(fig, rect=111, nrows_ncols=(5, 5), axes_pad=0.25, label_mode='L', share_all=True)
ax_cnt = 0
for city_1 in range(5):
    for city_2 in range(5):
        city_select = [i for i in range(5) if i != city_1 and i != city_2]
        gmm = train_gmm_source_domain(task_dir, idx, feature, truth_city, truth_building, city_select, n_comp,
                                      force_run=False, only_building=only_building)
        llh = test_gmm_model(idx, patch_names, gmm, feature)
        str2show = 'No {}'.format(city_list[city_2])
        ylab2show = 'No {}'.format(city_list[city_1])
        if city_1 == 0:
            plot_llh(llh, city_select, ax=grid[ax_cnt], title=str2show, ylab=ylab2show)
        else:
            plot_llh(llh, city_select, ax=grid[ax_cnt], ylab=ylab2show)
        ax_cnt += 1
plt.tight_layout()
if only_building:
    plt.savefig(os.path.join(img_dir, 'similarity_cmp_n{}_building.png'.format(n_comp)))
else:
    plt.savefig(os.path.join(img_dir, 'similarity_cmp_n{}_all.png'.format(n_comp)))
plt.show()'''


# train GMM model on test set
llh_all = np.zeros((5, 5))
fig = plt.figure(figsize=(12, 4))
grid = Grid(fig, rect=111, nrows_ncols=(1, 5), axes_pad=0.25, label_mode='L', share_all=True)
for i in range(5):
    city_select = [i]
    gmm = train_gmm(task_dir, np.array(idx) < 6, feature, truth_city, truth_building, city_select, n_comp,
                    force_run=False, only_building=False)
    llh, bic = test_gmm_model(idx, patch_names, gmm, feature, test_select=np.array(idx) >= 6, use_bic=True)
    llh_all[i, :] = llh
    t = (-bic/150) ** 3
    plot_llh(llh, city_select, title='{} score={:.3e}'.format(city_list[i], bic), t=t, ax=grid[i])
plt.tight_layout()
#plt.savefig(os.path.join(img_dir, 'similarity_cmp_n{}_all_train6_adjust.png'.format(n_comp)))
plt.show()
np.save(os.path.join(task_dir, 'llh_unet_inria_n{}.npy'.format(n_comp)), llh_all)


# test GMM score when loo
'''fig = plt.figure(figsize=(12, 4))
grid = Grid(fig, rect=111, nrows_ncols=(1, 5), axes_pad=0.25, label_mode='L', share_all=True)
for i in range(5):
    city_select = [i]
    gmm = train_gmm(task_dir, np.array(idx) < 6, feature, truth_city, truth_building, city_select, n_comp,
                    force_run=False, only_building=False)
    llh, bic = test_gmm_model(idx, patch_names, gmm, feature, test_select=np.array(idx) >= 6, use_bic=True,
                              test_city=[city_cnt for city_cnt in range(5) if city_cnt != i])
    plot_llh(llh, city_select, title='{} score={:.3e}'.format(city_list[i], bic), t=1500, ax=grid[i],
             x_pos=[city_cnt for city_cnt in range(5) if city_cnt != i])
plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'similarity_cmp_n{}_all_train6_loo.png'.format(n_comp)))
plt.show()'''

'''T = [5278, 1103, 48317, 11534, 1538]
for i in tqdm(range(5)):
    city_select = [i]
    gmm = train_gmm(task_dir, np.array(idx) < 6, feature, truth_city, truth_building, city_select, n_comp,
                    force_run=False, only_building=False)
    llh, train_idx = test_gmm_model_sample_wise(idx, gmm, feature, truth_city, range(5),
                                                test_select=np.array(idx) >= 6)
    plt.figure(figsize=(14, 6))
    plot_sample_wise_llh(llh, train_idx, truth_city, t=T[i])
    plt.title(city_list[i])
    plt.savefig(os.path.join(img_dir, 'similarity_cmp_n{}_sample_wise_{}_adjust.png'.format(n_comp, city_list[i])))
plt.show()'''
