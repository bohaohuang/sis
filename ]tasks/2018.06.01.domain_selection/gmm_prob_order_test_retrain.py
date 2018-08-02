import os
import imageio
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from mpl_toolkits.axes_grid1 import Grid
from sklearn.mixture import GaussianMixture
import utils
import uabCrossValMaker
import uabPreprocClasses
import uab_collectionFunctions
import uab_DataHandlerFunctions
import bohaoCustom.uabPreprocClasses as bPreproc


# settings
city_list = ['austin', 'chicago', 'kitsap', 'tyrol-w', 'vienna']
cnn_name = 'deeplab'

img_dir, task_dir = utils.get_task_img_folder()
n_comp = 70
llh_all = []


blCol = uab_collectionFunctions.uabCollection('inria')
opDetObj = bPreproc.uabOperTileDivide(255)          # inria GT has value 0 and 255, we map it back to 0 and 1
# [3] is the channel id of GT
rescObj = uabPreprocClasses.uabPreprocMultChanOp([], 'GT_Divide.tif', 'Map GT to (0, 1)', [3], opDetObj)
rescObj.run(blCol)
img_mean = blCol.getChannelMeans([0, 1, 2])         # get mean of rgb info

# extract patches
if cnn_name == 'deeplab':
    ps = 321
    extrObj = uab_DataHandlerFunctions.uabPatchExtr([0, 1, 2, 4],
                                                    cSize=(ps, ps),
                                                    numPixOverlap=0,
                                                    extSave=['jpg', 'jpg', 'jpg', 'png'],
                                                    isTrain=True,
                                                    gtInd=3,
                                                    pad=0)
else:
    ps = 572
    extrObj = uab_DataHandlerFunctions.uabPatchExtr([0, 1, 2, 4],
                                                    cSize=(ps, ps),
                                                    numPixOverlap=184,
                                                    extSave=['jpg', 'jpg', 'jpg', 'png'],
                                                    isTrain=True,
                                                    gtInd=3,
                                                    pad=92)
patchDir = extrObj.run(blCol)
# get validation set
# use uabCrossValMaker to get fileLists for training and validation
idx, file_list = uabCrossValMaker.uabUtilGetFolds(patchDir, 'fileList.txt', 'force_tile')

model_name = 'res50'
# load patch names
patch_file = os.path.join(r'/media/ei-edl01/user/bh163/tasks/2018.05.22.evaluate_gan',
                          '{}_inria_2048_{}.txt'.format(model_name, cnn_name))
with open(patch_file, 'r') as f:
    patch_names = f.readlines()
feature_file = os.path.join(r'/media/ei-edl01/user/bh163/tasks/2018.05.22.evaluate_gan',
                            '{}_inria_2048_{}.csv'.format(model_name, cnn_name))
feature = pd.read_csv(feature_file, sep=',', header=None).values
# make truth
truth_file_city = os.path.join(task_dir, 'truth_inria_city_2048_{}.npy'.format(cnn_name))
if not os.path.exists(truth_file_city):
    print('Making ground truth city...')
    truth_city = np.zeros(len(patch_names))
    city_dict = {'austin': 0, 'chicago': 1, 'kitsap': 2, 'tyrol-w': 3, 'vienna': 4}
    for cnt, file in enumerate(tqdm(patch_names)):
        city_name = ''.join([i for i in file.split('_')[0] if not i.isdigit()])
        truth_city[cnt] = city_dict[city_name]
    np.save(truth_file_city, truth_city)
else:
    truth_city = np.load(truth_file_city)

# make truth
truth_file_building = os.path.join(task_dir, 'truth_inria_building_2048_{}.npy'.format(cnn_name))
if not os.path.exists(truth_file_building):
    print('Making ground truth building...')
    truth_building = np.zeros(len(patch_names))
    for cnt, file in enumerate(tqdm(patch_names)):
        gt_name = os.path.join(patchDir, '{}_GT_Divide.png'.format(file[:-1]))
        gt = imageio.imread(gt_name)
        portion = np.sum(gt) / (ps * ps)
        if portion > 0.2:
            truth_building[cnt] = 1
    np.save(truth_file_building, truth_building)
else:
    truth_building = np.load(truth_file_building)

# fit on training set
idx = np.array(idx)
truth_city_train = truth_city[idx >= 6]
feature_train = feature[idx >= 6, :]

for city_1 in range(5):
    city_select = [i for i in range(5) if i == city_1]
    train_idx = []
    for s in city_select:
        train_idx.append(truth_city_train == s)
    train_idx = np.any(train_idx, axis=0)
    train_idx = np.all([train_idx, truth_building[idx >= 6] == 1], axis=0)
    gmm_models = GaussianMixture(n_components=n_comp, covariance_type='diag')
    gmm_models.fit(feature_train[train_idx, :])

    # compute llh for each patch
    city_dict = {'aus': 0, 'chi': 1, 'kit': 2, 'tyr': 3, 'vie': 4}
    feature_valid = feature[idx < 6, :]
    patch_valid = [patch_names[i] for i in range(len(patch_names)) if idx[i] < 6]
    city_name_list = [a[:3] for a in patch_valid]
    city_id_list = [city_dict[a] for a in city_name_list]

    fig = plt.figure(figsize=(3, 8))
    grid = Grid(fig, rect=111, nrows_ncols=(5, 1), axes_pad=0.25, label_mode='L', share_all=True)
    for test_city in range(5):
        test_city_feature = feature_valid[[i for i in range(len(city_id_list)) if
                                           city_id_list[i] == test_city], :]

        llh = gmm_models.score_samples(test_city_feature)

        ax = grid[test_city]
        ax.hist(llh, bins=100)
        ax.set_xlabel('LLH')
        ax.set_ylabel(city_list[test_city])
    plt.suptitle(city_list[city_1])
    plt.tight_layout()
    #plt.savefig(os.path.join(img_dir, 'llh_hist_1_no_{}.png'.
    #                         format('_'.join([city_list[a] for a in range(5) if a not in city_select]))))
    plt.show()
    #plt.close(fig)
