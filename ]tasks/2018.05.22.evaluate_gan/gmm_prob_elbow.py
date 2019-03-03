import os
import imageio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.mixture import GaussianMixture
import sis_utils
import uabCrossValMaker
import uabPreprocClasses
import uab_collectionFunctions
import uab_DataHandlerFunctions
import bohaoCustom.uabPreprocClasses as bPreproc


# settings
city_list = ['austin', 'chicago', 'kitsap', 'tyrol-w', 'vienna']

img_dir, task_dir = sis_utils.get_task_img_folder()
blCol = uab_collectionFunctions.uabCollection('inria')
opDetObj = bPreproc.uabOperTileDivide(255)          # inria GT has value 0 and 255, we map it back to 0 and 1
# [3] is the channel id of GT
rescObj = uabPreprocClasses.uabPreprocMultChanOp([], 'GT_Divide.tif', 'Map GT to (0, 1)', [3], opDetObj)
rescObj.run(blCol)
img_mean = blCol.getChannelMeans([0, 1, 2])         # get mean of rgb info

# extract patches
extrObj = uab_DataHandlerFunctions.uabPatchExtr([0, 1, 2, 4],
                                                cSize=(321, 321),
                                                numPixOverlap=0,
                                                extSave=['jpg', 'jpg', 'jpg', 'png'],
                                                isTrain=True,
                                                gtInd=3,
                                                pad=0)
patchDir = extrObj.run(blCol)
# get validation set
# use uabCrossValMaker to get fileLists for training and validation
idx, file_list = uabCrossValMaker.uabUtilGetFolds(patchDir, 'fileList.txt', 'force_tile')

model_name = 'res50'
# load patch names
patch_file = os.path.join(r'/media/ei-edl01/user/bh163/tasks/2018.06.01.domain_selection',
                          '{}_inria_2048.txt'.format(model_name))
with open(patch_file, 'r') as f:
    patch_names = f.readlines()
# make truth
truth_file_building = os.path.join(task_dir, 'truth_inria_building_2048.npy')
if not os.path.exists(truth_file_building):
    print('Making ground truth building...')
    truth_building = np.zeros(len(patch_names))
    for cnt, file in enumerate(tqdm(patch_names)):
        gt_name = os.path.join(patchDir, '{}_GT_Divide.png'.format(file[:-1]))
        gt = imageio.imread(gt_name)
        portion = np.sum(gt) / (321 * 321)
        if portion > 0.2:
            truth_building[cnt] = 1
    np.save(truth_file_building, truth_building)
else:
    truth_building = np.load(truth_file_building)
truth_file_city = os.path.join(task_dir, 'truth_inria_city_2048.npy')
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

# load features
feature_file = os.path.join(r'/media/ei-edl01/user/bh163/tasks/2018.06.01.domain_selection',
                            '{}_inria_2048.csv'.format(model_name))
feature = pd.read_csv(feature_file, sep=',', header=None).values

# fit on training set
llh_curve = []
llh_curve_test = []
test_points = list(range(10, 51, 5)) + list(range(50, 151, 20)) + list(range(180, 301, 30))
'''list(range(50, 151, 20)) + list(range(180, 301, 30)) + list(range(350, 1001, 50)) + \
              list(range(1100, 1501, 100))'''
curve_points = [i for i in test_points]
for n_comp in test_points:
    print('N Comp = {}'.format(n_comp))
    idx = np.array(idx)
    truth_city_train = truth_city[idx >= 6]
    feature_train = feature[idx >= 6, :]
    truth_building_train = truth_building[idx >= 6]
    gmm_models = []
    model_file_name = os.path.join(task_dir, 'gmm_models_{}_{}_2048.npy'.format(model_name, n_comp))
    if not os.path.exists(model_file_name):
        print('\ttrain GMM models ...')
        for i in tqdm(range(5)):
            gmm = GaussianMixture(n_components=n_comp, covariance_type='diag')
            gmm.fit(feature_train[truth_city_train == i, :])
            gmm_models.append(gmm)
            np.save(model_file_name, gmm_models)
    else:
        print('loading models')
        gmm_models = np.load(model_file_name)

    # evaluate on train set
    idx = np.array(idx)
    track_id = np.arange(len(patch_names))
    track_id_valid = track_id[idx >= 6]
    truth_city_valid = truth_city[idx >= 6]
    feature_valid = feature[idx >= 6, :]
    truth_building_valid = truth_building[idx >= 6]

    track_id_test = track_id[idx < 6]
    truth_city_test = truth_city[idx < 6]
    feature_test = feature[idx < 6, :]
    truth_building_test = truth_building[idx < 6]

    llh_model = []
    llh_model_test = []
    for i in range(5):
        print('\tevaluating city {}'.format(city_list[i]))
        for j in range(5):
            llh = gmm_models[j].score_samples(feature_valid[truth_city_valid == i, :])
            llh_model.append(llh)
            llh = gmm_models[j].score_samples(feature_test[truth_city_test == i, :])
            llh_model_test.append(llh)
    llh_curve.append(np.mean(llh_model))
    llh_curve_test.append(np.mean(llh_model_test))

plt.figure(figsize=(10, 8))
plt.subplot(211)
plt.plot(curve_points, llh_curve)
plt.xlabel('')
plt.ylabel('LLH Train')
plt.grid(True)
plt.subplot(212)
plt.plot(curve_points, llh_curve_test)
plt.xlabel('N Comp')
plt.ylabel('LLH Valid')
plt.grid(True)
plt.tight_layout()
# plt.savefig(os.path.join(img_dir, 'elbow_{}_2048.png'.format(model_name)))
plt.show()
