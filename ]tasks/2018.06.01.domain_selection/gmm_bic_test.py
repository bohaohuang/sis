import os
import imageio
import pickle
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
cnn_name = 'deeplab'

img_dir, task_dir = utils.get_task_img_folder()
save_file_name = os.path.join(task_dir, 'llh_bic_test.npy')
force_run = False
test_points = list(range(10, 151, 10)) + list(range(160, 501, 20))

if not os.path.exists(save_file_name) or force_run:
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

    pbar = tqdm(test_points)
    bic = np.zeros(len(test_points))
    for cnt, n_comp in enumerate(pbar):
        pbar.set_description('Training GMM n_comp={}'.format(n_comp))
        train_idx = []
        gmm_models = GaussianMixture(n_components=n_comp, covariance_type='diag')
        gmm_models.fit(feature_train[truth_building[idx >= 6] == 1, :])
        bic[cnt] = gmm_models.bic(feature[idx < 6, :])
    np.save(save_file_name, bic)
else:
    bic = np.load(save_file_name)

plt.plot(test_points, bic)
plt.show()
