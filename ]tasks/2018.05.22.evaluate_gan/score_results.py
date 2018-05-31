import os
import itertools
import imageio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc, confusion_matrix
import utils
import uabCrossValMaker
import uabPreprocClasses
import uab_collectionFunctions
import uab_DataHandlerFunctions
import bohaoCustom.uabPreprocClasses as bPreproc


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

plt.figure(figsize=(8, 6))
fig_num = plt.gcf().number
for model_name in ['res50', 'vae', 'ali']:
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

    # do on valid set
    idx = np.array(idx)
    truth_building = truth_building[idx < 6]
    truth_city = truth_city[idx < 6]
    feature = feature[idx < 6, :]

    # do cross validation
    np.random.seed(1004)

    pred_file_name = os.path.join(task_dir, '{}_building_pred.npy'.format(model_name))
    if not os.path.exists(pred_file_name):
        kf = KFold(n_splits=5)
        clf = svm.SVC(probability=True)
        pred_building = []
        truth_building_rearrange = []
        for cnt, (train_idx, test_idx) in enumerate(kf.split(feature)):
            print('Training on fold {}'.format(cnt))
            X_train, X_test = feature[train_idx, :], feature[test_idx, :]
            y_train, y_test = truth_building[train_idx], truth_building[test_idx]
            clf.fit(X_train, y_train)
            pred_building.append(clf.predict_proba(X_test)[:,1])
            truth_building_rearrange.append(y_test)
        pred_building = np.concatenate(pred_building)
        truth_building_rearrange = np.concatenate(truth_building_rearrange)
        np.save(pred_file_name, [pred_building, truth_building_rearrange])
    else:
        pred_building, truth_building_rearrange = np.load(pred_file_name)

    pred_file_name = os.path.join(task_dir, '{}_city_pred.npy'.format(model_name))
    if not os.path.exists(pred_file_name):
        kf = KFold(n_splits=5, shuffle=True)
        clf = OneVsRestClassifier(svm.SVC(probability=True))
        pred_city = []
        truth_city_rearrange = []
        for cnt, (train_idx, test_idx) in enumerate(kf.split(feature)):
            print('Training on fold {}'.format(cnt))
            X_train, X_test = feature[train_idx, :], feature[test_idx, :]
            y_train, y_test = truth_city[train_idx], truth_city[test_idx]
            clf.fit(X_train, y_train)
            pred_city.append(clf.predict(X_test))
            truth_city_rearrange.append(y_test)
        pred_city = np.concatenate(pred_city)
        truth_city_rearrange = np.concatenate(truth_city_rearrange)
        np.save(pred_file_name, [pred_city, truth_city_rearrange])
    else:
        pred_city, truth_city_rearrange = np.load(pred_file_name)

    plt.figure(fig_num)
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
    plot_confusion_matrix(cnf_matrix, classes=['austin', 'chicago', 'kitsap', 'tyrol-w', 'vienna'], normalize=True,
                          title='CM City ({})'.format(model_name))
plt.show()
