import os
import imageio
import itertools
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc, confusion_matrix
from mpl_toolkits.axes_grid1 import Grid
import utils
import util_functions
import uabCrossValMaker
import uabPreprocClasses
import uab_collectionFunctions
import uab_DataHandlerFunctions
import bohaoCustom.uabPreprocClasses as bPreproc
from bohaoCustom import uabMakeNetwork_DeepLabV2
from bohaoCustom import uabMakeNetwork_UNet


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# settings
img_dir, task_dir = utils.get_task_img_folder()
np.random.seed(1004)
model_name = 'unet'
verify = False
force_run = True

# make network
input_size_fit = (299, 299)
if model_name == 'deeplab':
    input_size = (321, 321)
    # define place holder
    X = tf.placeholder(tf.float32, shape=[None, input_size[0], input_size[1], 3], name='X')
    y = tf.placeholder(tf.int32, shape=[None, input_size[0], input_size[1], 1], name='y')
    mode = tf.placeholder(tf.bool, name='mode')
    model = uabMakeNetwork_DeepLabV2.DeeplabV3({'X':X, 'Y':y},
                                               trainable=mode,
                                               input_size=input_size,)
else:
    input_size = (572, 572)
    # define place holder
    X = tf.placeholder(tf.float32, shape=[None, input_size[0], input_size[1], 3], name='X')
    y = tf.placeholder(tf.int32, shape=[None, input_size[0], input_size[1], 1], name='y')
    mode = tf.placeholder(tf.bool, name='mode')
    model = uabMakeNetwork_UNet.UnetModelCrop({'X': X, 'Y': y},
                                              trainable=mode,
                                              input_size=input_size, )

blCol = uab_collectionFunctions.uabCollection('inria')
opDetObj = bPreproc.uabOperTileDivide(255)          # inria GT has value 0 and 255, we map it back to 0 and 1
# [3] is the channel id of GT
rescObj = uabPreprocClasses.uabPreprocMultChanOp([], 'GT_Divide.tif', 'Map GT to (0, 1)', [3], opDetObj)
rescObj.run(blCol)
img_mean = blCol.getChannelMeans([0, 1, 2])         # get mean of rgb info

# extract patches
extrObj = uab_DataHandlerFunctions.uabPatchExtr([0, 1, 2, 4],
                                                cSize=(input_size[0], input_size[1]),
                                                numPixOverlap=int(model.get_overlap()),
                                                extSave=['jpg', 'jpg', 'jpg', 'png'],
                                                isTrain=True,
                                                gtInd=3,
                                                pad=model.get_overlap()/2)
patchDir = extrObj.run(blCol)
# get validation set
# use uabCrossValMaker to get fileLists for training and validation
idx, file_list = uabCrossValMaker.uabUtilGetFolds(patchDir, 'fileList.txt', 'force_tile')

# load patch names
patch_file = os.path.join(task_dir, 'incep_inria_{}.txt'.format(model_name))
with open(patch_file, 'r') as f:
    patch_names = f.readlines()
# make truth
truth_file_building = os.path.join(task_dir, 'truth_inria_building_{}.npy'.format(model_name))
truth_file_buidling10 = os.path.join(task_dir, 'truth_inria_building10_{}.npy'.format(model_name))
if (not os.path.exists(truth_file_building)) or (not os.path.exists(truth_file_buidling10)):
    print('Making ground truth building...')
    truth_building = np.zeros(len(patch_names))
    truth_building10 = np.zeros(len(patch_names))
    for cnt, file in enumerate(tqdm(patch_names)):
        gt_name = os.path.join(patchDir, '{}_GT_Divide.png'.format(file[:-1]))
        gt = imageio.imread(gt_name)
        if model_name == 'unet':
            portion = np.sum(util_functions.crop_center(gt, 388, 388)) / (388 * 388)
        else:
            portion = np.sum(gt) / (321 * 321)
        if portion > 0.1:
            truth_building[cnt] = 1
        if portion // 0.1 <= 5:
            truth_building10[cnt] = portion // 0.1
        else:
            truth_building10[cnt] = 6
    np.save(truth_file_building, truth_building)
    np.save(truth_file_buidling10, truth_building10)
else:
    truth_building = np.load(truth_file_building)
    truth_building10 = np.load(truth_file_buidling10)

# verify gt
if verify:
    fig = plt.figure(figsize=(12, 8))
    grid = Grid(fig, rect=111, nrows_ncols=(4, 5), axes_pad=0.25, label_mode='L')
    rand_idx = np.random.permutation(len(patch_names))
    rand_patch_names = [patch_names[rand_i] for rand_i in rand_idx]
    for cnt, file in enumerate(rand_patch_names[:20]):
        gt_name = os.path.join(patchDir, '{}_GT_Divide.png'.format(file[:-1]))
        gt_img = imageio.imread(gt_name)
        gt_building = truth_building10[rand_idx[cnt]]
        grid[cnt].imshow(gt_img)
        grid[cnt].set_title(gt_building)
    plt.show()


# load features
feature_file = os.path.join(task_dir, 'incep_inria_{}.csv'.format(model_name))
feature = pd.read_csv(feature_file, sep=',', header=None).values
assert len(idx) == feature.shape[0]

# do on valid set
idx = np.array(idx)
truth_building = truth_building[idx < 6]
truth_building10 = truth_building10[idx < 6]
feature = feature[idx < 6, :]

# do cross validation
pred_file_name = os.path.join(task_dir, 'incep_building_pred_{}.npy'.format(model_name))
if (not os.path.exists(pred_file_name)) or force_run:
    kf = KFold(n_splits=5, shuffle=True)
    clf = svm.SVC(probability=True)
    pred_building = []
    truth_building_rearrange = []
    pbar = tqdm(kf.split(feature))
    for cnt, (train_idx, test_idx) in enumerate(pbar):
        pred_file_name_seperate = os.path.join(task_dir, 'incep_building_pred_{}_loo{}.npy'.format(model_name, cnt))
        pbar.set_description('Training building on fold {}'.format(cnt))
        X_train, X_test = feature[train_idx, :], feature[test_idx, :]
        y_train, y_test = truth_building[train_idx], truth_building[test_idx]
        clf.fit(X_train, y_train)
        pred = clf.predict_proba(X_test)[:,1]
        pred_building.append(pred)
        truth_building_rearrange.append(y_test)
        np.save(pred_file_name_seperate, [pred, y_test])
    pred_building = np.concatenate(pred_building)
    truth_building_rearrange = np.concatenate(truth_building_rearrange)
    np.save(pred_file_name, [pred_building, truth_building_rearrange])
else:
    pred_building, truth_building_rearrange = np.load(pred_file_name)

pred_file_name = os.path.join(task_dir, 'incep_building10_pred_{}.npy'.format(model_name))
if (not os.path.exists(pred_file_name)) or force_run:
    kf = KFold(n_splits=5, shuffle=True)
    clf = OneVsRestClassifier(svm.SVC(probability=True))
    pred_city = []
    truth_building10_rearrange = []
    pbar = kf.split(feature)
    for cnt, (train_idx, test_idx) in enumerate(pbar):
        pred_file_name_seperate = os.path.join(task_dir, 'incep_building10_pred_{}_loo{}.npy'.format(model_name, cnt))
        pbar.set_description('Training building10 on fold {}'.format(cnt))
        X_train, X_test = feature[train_idx, :], feature[test_idx, :]
        y_train, y_test = truth_building10[train_idx], truth_building10[test_idx]
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        pred_city.append(pred)
        truth_building10_rearrange.append(y_test)
        np.save(pred_file_name, [pred_building, truth_building_rearrange])
    pred_city = np.concatenate(pred_city)
    truth_city_rearrange = np.concatenate(truth_building10_rearrange)
    np.save(pred_file_name, [pred_city, truth_city_rearrange])
else:
    pred_city, truth_city_rearrange = np.load(pred_file_name)

plt.figure()
fpr_rf, tpr_rf, _ = roc_curve(truth_building_rearrange, pred_building)
plt.plot(fpr_rf, tpr_rf, label='{} AUC = {:.2f}'.format(model_name, auc(fpr_rf, tpr_rf)))
plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Building')
plt.legend(loc="lower right")

plt.figure()
cnf_matrix = confusion_matrix(truth_city_rearrange, pred_city)
np.set_printoptions(precision=2)
plot_confusion_matrix(cnf_matrix,
                      classes=['0~10', '10~20', '20~30', '30~40', '40~50', '50~60', '60~100'], normalize=True,
                      title='CM Building10 ({})'.format(model_name))
plt.show()
