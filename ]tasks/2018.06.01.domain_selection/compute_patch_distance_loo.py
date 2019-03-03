import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
import sklearn.metrics
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
cnn_name = 'deeplab'

img_dir, task_dir = sis_utils.get_task_img_folder()
blCol = uab_collectionFunctions.uabCollection('inria')
opDetObj = bPreproc.uabOperTileDivide(255)          # inria GT has value 0 and 255, we map it back to 0 and 1
# [3] is the channel id of GT
rescObj = uabPreprocClasses.uabPreprocMultChanOp([], 'GT_Divide.tif', 'Map GT to (0, 1)', [3], opDetObj)
rescObj.run(blCol)
img_mean = blCol.getChannelMeans([0, 1, 2])         # get mean of rgb info

# extract patches
if cnn_name == 'deeplab':
    extrObj = uab_DataHandlerFunctions.uabPatchExtr([0, 1, 2, 4],
                                                    cSize=(321, 321),
                                                    numPixOverlap=0,
                                                    extSave=['jpg', 'jpg', 'jpg', 'png'],
                                                    isTrain=True,
                                                    gtInd=3,
                                                    pad=0)
else:
    extrObj = uab_DataHandlerFunctions.uabPatchExtr([0, 1, 2, 4],
                                                    cSize=(572, 572),
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

# fit on training set
idx = np.array(idx)
truth_city_train = truth_city[idx >= 6]
feature_train = feature[idx >= 6, :]

if cnn_name == 'deeplab':
    n_comp = 40
else:
    n_comp = 35
gmm_models = []
model_file_name = os.path.join(r'/media/ei-edl01/user/bh163/tasks/2018.05.22.evaluate_gan',
                               'gmm_models_{}_{}_2048_{}.npy'.format(model_name, n_comp, cnn_name))
if not os.path.exists(model_file_name):
    print('train GMM models ...')
    for i in tqdm(range(5)):
        gmm = GaussianMixture(n_components=n_comp, covariance_type='diag')
        gmm.fit(feature_train[truth_city_train == i, :])
        gmm_models.append(gmm)
        np.save(model_file_name, gmm_models)
else:
    #print('loading models')
    gmm_models = np.load(model_file_name)

# compute llh for each patch
city_dict = {'aus': 0, 'chi': 1, 'kit': 2, 'tyr': 3, 'vie': 4}
feature_valid = feature[idx < 6, :]
patch_valid = [patch_names[i] for i in range(len(patch_names)) if idx[i] < 6]
city_name_list = [a[:3] for a in patch_valid]
city_id_list = [city_dict[a] for a in city_name_list]

fig = plt.figure(figsize=(12, 6))
prior_file_name = os.path.join(task_dir, 'prior_gmm_n_comp_{}_LOO.txt'.format(n_comp))
llh_all = np.zeros((5, 5))
for test_city in range(5):
    test_city_feature = feature_valid[[i for i in range(len(city_id_list)) if city_id_list[i] == test_city], :]

    llh = np.zeros(4)
    for llh_cnt, i in enumerate([x for x in range(5) if x != test_city]):
        llh[llh_cnt] = gmm_models[i].score(test_city_feature)

    llh = (llh - np.min(llh)) / (np.max(llh) - np.min(llh)) + 0.1
    llh = llh / np.sum(llh)
    for llh_cnt, i in enumerate([x for x in range(5) if x != test_city]):
        llh_all[test_city][i] = llh[llh_cnt]

    plt.subplot(231 + test_city)
    X = np.arange(4)
    width = 0.5
    plt.bar(X, llh)
    for cnt, val in enumerate(llh):
        plt.text(X[cnt] - width*0.48, val, '{:.3f}'.format(val))
    plt.xticks(X, [city_list[x] for x in range(5) if x != test_city])
    plt.title('{} LOO'.format(city_list[test_city]))
plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'llh_c_num{}_{}_LOO_patchwise.png'.format(n_comp, cnn_name)))
plt.show()

'''with open(prior_file_name, 'w') as f:
    for test_city in range(5):
        f.write('{}\n'.format(','.join([str(num) for num in llh_all[test_city]])))'''

