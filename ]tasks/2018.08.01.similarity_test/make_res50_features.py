"""
Remake Res50 features for Inria dataset
Calm down, do it step by step, make sure it is bug-free
"""


import os
import csv
import imageio
import numpy as np
import bohaoCustom.uabPreprocClasses as bPreproc
import uabPreprocClasses
import uabCrossValMaker
import uab_collectionFunctions
import uab_DataHandlerFunctions
from tqdm import tqdm
import utils


def crop_center(img,cropx,cropy):
    y,x, _ = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[starty:starty+cropy,startx:startx+cropx, :]


def make_res50_features(model_name, task_dir, GPU=0, force_run=False):
    feature_file_name = os.path.join(task_dir, 'res50_inria_{}.csv'.format(model_name))
    patch_file_name = os.path.join(task_dir, 'res50_inria_{}.txt'.format(model_name))

    if model_name == 'deeplab':
        input_size = (321, 321)
        overlap = 0
    else:
        input_size = (572, 572)
        overlap = 184
    blCol = uab_collectionFunctions.uabCollection('inria')
    opDetObj = bPreproc.uabOperTileDivide(255)
    rescObj = uabPreprocClasses.uabPreprocMultChanOp([], 'GT_Divide.tif', 'Map GT to (0, 1)', [3], opDetObj)
    rescObj.run(blCol)
    img_mean = blCol.getChannelMeans([0, 1, 2])
    extrObj = uab_DataHandlerFunctions.uabPatchExtr([0, 1, 2, 4],
                                                    cSize=input_size,
                                                    numPixOverlap=overlap,
                                                    extSave=['jpg', 'jpg', 'jpg', 'png'],
                                                    isTrain=True,
                                                    gtInd=3,
                                                    pad=overlap // 2)
    patchDir = extrObj.run(blCol)
    idx, _ = uabCrossValMaker.uabUtilGetFolds(patchDir, 'fileList.txt', 'force_tile')

    if not os.path.exists(feature_file_name) or not os.path.exists(patch_file_name) or force_run:
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(GPU)
        import keras

        input_size_fit = (224, 224)

        file_name = os.path.join(patchDir, 'fileList.txt')
        with open(file_name, 'r') as f:
            files = f.readlines()

        res50 = keras.applications.resnet50.ResNet50(include_top=True, weights='imagenet')
        fc2048 = keras.models.Model(inputs=res50.input, outputs=res50.get_layer('flatten_1').output)
        with open(feature_file_name, 'w+') as f:
            with open(patch_file_name, 'w+') as f2:
                for file_line in tqdm(files):
                    patch_name = file_line.split('.')[0][:-5]
                    img = np.zeros((input_size[0], input_size[1], 3), dtype=np.uint8)
                    for cnt, file in enumerate(file_line.strip().split(' ')[:3]):
                        img[:, :, cnt] = imageio.imread(os.path.join(patchDir, file)) - img_mean[cnt]

                    img = np.expand_dims(crop_center(img, input_size_fit[0], input_size_fit[1]), axis=0)

                    fc1000 = fc2048.predict(img).reshape((-1,)).tolist()
                    writer = csv.writer(f, lineterminator='\n')
                    writer.writerow(['{}'.format(x) for x in fc1000])
                    f2.write('{}\n'.format(patch_name))

    return feature_file_name, patch_file_name, input_size[0], patchDir, idx


def check_res50_features(model_name, GPU=0):
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(GPU)
    import keras

    input_size_fit = (224, 224)

    blCol = uab_collectionFunctions.uabCollection('inria')
    opDetObj = bPreproc.uabOperTileDivide(255)
    rescObj = uabPreprocClasses.uabPreprocMultChanOp([], 'GT_Divide.tif', 'Map GT to (0, 1)', [3], opDetObj)
    rescObj.run(blCol)
    img_mean = blCol.getChannelMeans([0, 1, 2])

    if model_name == 'deeplab':
        input_size = (321, 321)
        overlap = 0
    else:
        input_size = (572, 572)
        overlap = 184
    extrObj = uab_DataHandlerFunctions.uabPatchExtr([0, 1, 2, 4],
                                                    cSize=input_size,
                                                    numPixOverlap=overlap,
                                                    extSave=['jpg', 'jpg', 'jpg', 'png'],
                                                    isTrain=True,
                                                    gtInd=3,
                                                    pad=overlap // 2)
    patchDir = extrObj.run(blCol)

    file_name = os.path.join(patchDir, 'fileList.txt')
    with open(file_name, 'r') as f:
        files = f.readlines()

    res50 = keras.applications.resnet50.ResNet50(include_top=True, weights='imagenet')
    pred_list = np.zeros(len(files))
    for file_cnt, file_line in enumerate(tqdm(files)):
        img = np.zeros((input_size[0], input_size[1], 3), dtype=np.uint8)
        for cnt, file in enumerate(file_line.strip().split(' ')[:3]):
            img[:, :, cnt] = imageio.imread(os.path.join(patchDir, file)) - img_mean[cnt]

        img = np.expand_dims(crop_center(img, input_size_fit[0], input_size_fit[1]), axis=0)

        fc1000 = res50.predict(img).reshape((-1,)).tolist()
        pred_list[file_cnt] = np.argmax(fc1000)
    return pred_list


if __name__ == '__main__':
    img_dir, task_dir = utils.get_task_img_folder()
    feature_file_name, patch_file_name = \
        make_res50_features('deeplab', task_dir, GPU=0, force_run=False)

    pred_list = check_res50_features('deeplab', GPU=0)
    print(pred_list)
    import matplotlib.pyplot as plt
    plt.hist(pred_list)
    plt.show()
