import os
import imageio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.mixture import GaussianMixture
import utils
import uabCrossValMaker
import uabPreprocClasses
import uab_collectionFunctions
import uab_DataHandlerFunctions
import bohaoCustom.uabPreprocClasses as bPreproc


# settings
city_list = ['austin', 'chicago', 'kitsap', 'tyrol-w', 'vienna']

img_dir, task_dir = utils.get_task_img_folder()
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

model_name = 'vae'
# load patch names
patch_file = os.path.join(task_dir, '{}_inria.txt'.format(model_name))
with open(patch_file, 'r') as f:
    patch_names = f.readlines()
# make truth
truth_file_building = os.path.join(task_dir, 'truth_inria_building.npy')
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
truth_file_city = os.path.join(task_dir, 'truth_inria_city.npy')
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
feature_file = os.path.join(task_dir, '{}_inria.csv'.format(model_name))
feature = pd.read_csv(feature_file, sep=',', header=None).values

# fit on training set
n_comp = 80
idx = np.array(idx)
truth_city_train = truth_city[idx >= 6]
feature_train = feature[idx >= 6, :]
truth_building_train = truth_building[idx >= 6]
gmm_models = []
model_file_name = os.path.join(task_dir, 'gmm_models_{}_{}.npy'.format(model_name, n_comp))
if not os.path.exists(model_file_name):
    print('train GMM models ...')
    for i in tqdm(range(5)):
        gmm = GaussianMixture(n_components=n_comp, covariance_type='diag')
        #gmm.fit(feature_train[np.all([truth_city_train == i, truth_building_train == 1], axis=0), :])
        gmm.fit(feature_train[truth_city_train == i, :])
        gmm_models.append(gmm)
        np.save(model_file_name, gmm_models)
else:
    print('loading models')
    gmm_models = np.load(model_file_name)

# evaluate on valid set
idx = np.array(idx)
track_id = np.arange(len(patch_names))
track_id_valid = track_id[idx < 6]
truth_city_valid = truth_city[idx < 6]
feature_valid = feature[idx < 6, :]
truth_building_valid = truth_building[idx < 6]
for i in range(5):
    print('evaluating city {}'.format(city_list[i]))
    plt.figure(figsize=(6, 12))
    for j in range(5):
        llh = gmm_models[j].score_samples(feature_valid[truth_city_valid == i, :])
        #tid = track_id_valid[np.all([truth_city_valid == i, truth_building_valid == 1], axis=0)]
        tid = track_id_valid[truth_city_valid == i]
        top5 = tid[np.argsort(llh)[::-1][:5]]

        img = np.zeros((321, 321*5, 3), dtype=np.uint8)
        for img_cnt, pid in enumerate(top5):
            for c_cnt in range(3):
                img[:, 321*img_cnt:321*(img_cnt+1), c_cnt] = imageio.imread(
                    os.path.join(patchDir, '{}_RGB{}.jpg'.format(patch_names[pid][:-1], c_cnt)))
        imageio.imsave(os.path.join(img_dir, '{}_top5_{}_model_on_{}.png'.format(model_name, city_list[j], city_list[i])),
                       img)

        if j == 0:
            ax = plt.subplot(511 + j)
        else:
            plt.subplot(511 + j, sharex=ax, sharey=ax)
        plt.hist(llh, bins=100)
        plt.axvline(x=np.mean(llh), color='r', linestyle='--', linewidth=2)
        plt.title('{} model on {} (Avg: {:.3f})'.format(city_list[j], city_list[i], np.mean(llh)))
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, '{}_{}.png'.format(model_name, city_list[i])))
plt.show()
