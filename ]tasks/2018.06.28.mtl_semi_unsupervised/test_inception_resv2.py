import os
import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import sis_utils
import uabCrossValMaker
import uab_collectionFunctions
import uab_DataHandlerFunctions
import uabUtilreader
import util_functions
from bohaoCustom import uabDataReader


class ImageLabelReaderBuilding(uabDataReader.ImageLabelReader):
    def __init__(self, gtInds, dataInds, parentDir, chipFiles, chip_size, batchSize, center_crop, patch_prob,
                 nChannels=1, padding=np.array((0, 0)), block_mean=None, dataAug=''):
        self.patch_prob = patch_prob
        self.center_crop = center_crop
        super(ImageLabelReaderBuilding, self).__init__(gtInds, dataInds, parentDir,
                                                       chipFiles,
                                                       chip_size,
                                                       batchSize, nChannels, padding,
                                                       block_mean, dataAug)

    def readFromDiskIteratorTrain(self, image_dir, chipFiles, batch_size, patch_size, padding=(0, 0), dataAug=''):
        # this is a iterator for training
        nDims = len(chipFiles[0])
        image_batch = np.zeros((batch_size, self.center_crop[0], self.center_crop[1], nDims))
        building_truth = np.zeros((batch_size, 2))
        # select number to sample
        idx_batch = np.random.permutation(len(chipFiles))
        for cnt, randInd in enumerate(idx_batch):
            row = chipFiles[randInd]

            blockList = []
            nDims = 0
            for file in row:
                img = util_functions.uabUtilAllTypeLoad(os.path.join(image_dir, file))
                if len(img.shape) == 2:
                    img = np.expand_dims(img, axis=2)
                nDims += img.shape[2]
                blockList.append(img)
            block = np.dstack(blockList).astype(np.float32)

            if self.block_mean is not None:
                block -= self.block_mean

            if dataAug != '':
                augDat = uabUtilreader.doDataAug(block, nDims, dataAug, is_np=True,
                                                 img_mean=self.block_mean)
            else:
                augDat = block

            if (np.array(padding) > 0).any():
                augDat = uabUtilreader.pad_block(augDat, padding)

            augDat = util_functions.crop_center(augDat, self.center_crop[0], self.center_crop[1])

            store_idx = cnt % batch_size
            image_batch[store_idx, :, :, :] = augDat
            percent = np.sum(augDat[:, :, 0]) / (self.center_crop[0] * self.center_crop[1])

            if percent > self.patch_prob:
                building_truth[store_idx, :] = [0, 1]
            else:
                building_truth[store_idx, :] = [1, 0]

            if (cnt + 1) % batch_size == 0:
                yield (image_batch[:, :, :, 1:], building_truth)


if __name__ == '__main__':
    city_list = ['austin', 'chicago', 'kitsap', 'tyrol-w', 'vienna']
    model_name = 'unet'
    leave_city = 1
    batch_size = 100
    prescr_name = 'incep'
    blCol = uab_collectionFunctions.uabCollection('inria')
    img_mean = blCol.getChannelMeans([0, 1, 2])
    img_dir, task_dir = sis_utils.get_task_img_folder()

    if prescr_name == 'incep':
        center_crop = (299, 299)
    elif prescr_name == 'res50':
        center_crop = (224, 224)
    else:
        raise KeyError

    model_save_dir = os.path.join(task_dir, '{}_building_loo_{}.hdf5'.format(prescr_name, leave_city))
    model = keras.models.load_model(model_save_dir)

    if model_name == 'unet':
        patch_size = (572, 572)
        overlap = 184
        pad = 92
    else:
        patch_size = (321, 321)
        overlap = 0
        pad = 0

    # extract patches
    extrObj = uab_DataHandlerFunctions.uabPatchExtr([0, 1, 2, 4],
                                                    cSize=patch_size,
                                                    numPixOverlap=overlap,
                                                    extSave=['jpg', 'jpg', 'jpg', 'png'],
                                                    isTrain=True,
                                                    gtInd=3,
                                                    pad=pad)
    patchDir = extrObj.run(blCol)
    chipFiles = os.path.join(patchDir, 'fileList.txt')
    idx, file_list = uabCrossValMaker.uabUtilGetFolds(patchDir, 'fileList.txt', 'city')
    idx2, _ = uabCrossValMaker.uabUtilGetFolds(patchDir, 'fileList.txt', 'force_tile')
    idx3 = [j * 10 + i for i, j in zip(idx, idx2)]

    plt.figure()
    for leave_city in range(5):
        filter_train = []
        filter_valid = []
        for i in range(5):
            for j in range(1, 37):
                if i == leave_city:
                    filter_valid.append(j * 10 + i)
                else:
                    filter_train.append(j * 10 + i)
        file_list_train = uabCrossValMaker.make_file_list_by_key(idx3, file_list, filter_train)
        file_list_valid = uabCrossValMaker.make_file_list_by_key(idx3, file_list, filter_valid)

        dataReader_valid = ImageLabelReaderBuilding(
            [3], [0, 1, 2], patchDir, file_list_valid, patch_size, batch_size, center_crop, 0.1,
            block_mean=np.append([0], img_mean)).readManager

        truth_building = []
        pred_building = []
        for img, gt in dataReader_valid:
            pred = model.predict_on_batch(img)
            truth_building.append(np.argmax(gt, axis=1))
            # pred_building.append(pred[:, 1])
            pred_building.append(np.argmax(pred, axis=1))

        truth_building = [item for sublist in truth_building for item in sublist]
        pred_building = [item for sublist in pred_building for item in sublist]
        fpr_rf, tpr_rf, _ = roc_curve(truth_building, pred_building)
        plt.plot(fpr_rf, tpr_rf, label='{} AUC = {:.2f}'.format(city_list[leave_city], auc(fpr_rf, tpr_rf)))
    plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Building LOO')
    plt.legend(loc="lower right")
    plt.show()
