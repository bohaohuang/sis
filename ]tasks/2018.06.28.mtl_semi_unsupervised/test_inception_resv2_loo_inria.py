import os
import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import sis_utils
import uabRepoPaths
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
    batch_size = 100
    prescr_name = 'res50'
    blCol = uab_collectionFunctions.uabCollection('inria')
    img_mean = blCol.getChannelMeans([0, 1, 2])
    img_dir, task_dir = sis_utils.get_task_img_folder()
    force_run = False

    if force_run:
        if prescr_name == 'incep':
            center_crop = (299, 299)
        elif prescr_name == 'res50':
            center_crop = (224, 224)
        else:
            raise KeyError

        truth_building = []
        pred_building = []
        pred_building_binary = []
        for city_num in range(5):
            model_save_dir = os.path.join(task_dir, '{}_building_loo_{}.hdf5'.format(prescr_name, city_num))
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
            filter_train = []
            filter_valid = []
            for i in range(5):
                for j in range(1, 37):
                    if i == city_num and j <= 5:
                        filter_valid.append(j * 10 + i)
                    elif i != city_num:
                        filter_train.append(j * 10 + i)
            file_list_valid = uabCrossValMaker.make_file_list_by_key(idx3, file_list, filter_valid)

            dataReader_valid = ImageLabelReaderBuilding(
                [3], [0, 1, 2], patchDir, file_list_valid, patch_size, batch_size, center_crop, 0.1,
                block_mean=np.append([0], img_mean)).readManager

            for img, gt in dataReader_valid:
                pred = model.predict_on_batch(img)
                truth_building.append(np.argmax(gt, axis=1))
                pred_building.append(pred[:, 1])
                pred_building_binary.append(np.argmax(pred, axis=1))
        truth_building = [item for sublist in truth_building for item in sublist]
        pred_building = [item for sublist in pred_building for item in sublist]
        pred_building_binary = [item for sublist in pred_building_binary for item in sublist]
        np.save(os.path.join(task_dir, '{}_truth_building.npy'.format(prescr_name)), truth_building)
        np.save(os.path.join(task_dir, '{}_pred_building.npy'.format(prescr_name)), pred_building)
        np.save(os.path.join(task_dir, '{}_pred_building_binary.npy'.format(prescr_name)), pred_building_binary)
    else:
        res50_truth_building = np.load(os.path.join(task_dir, 'res50_truth_building.npy'))
        res50_pred_building = np.load(os.path.join(task_dir, 'res50_pred_building.npy'))
        res50_pred_building_binary = np.load(os.path.join(task_dir, 'res50_pred_building_binary.npy'))

        incep_truth_building = np.load(os.path.join(task_dir, 'incep_truth_building.npy'))
        incep_pred_building = np.load(os.path.join(task_dir, 'incep_pred_building.npy'))
        incep_pred_building_binary = np.load(os.path.join(task_dir, 'incep_pred_building_binary.npy'))

        plt.figure()
        colors = util_functions.get_default_colors()

        fpr_rf, tpr_rf, _ = roc_curve(res50_truth_building, res50_pred_building)
        plt.plot(fpr_rf, tpr_rf, '--', label='Res50 AUC = {:.2f}'.format(auc(fpr_rf, tpr_rf)), color=colors[0])
        fpr_rf, tpr_rf, _ = roc_curve(incep_truth_building, incep_pred_building)
        plt.plot(fpr_rf, tpr_rf, '--', label='Incep AUC = {:.2f}'.format(auc(fpr_rf, tpr_rf)), color=colors[1])
        plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--')

        plt.plot([0.028234771009337484], [0.8338590956887487], color=colors[2], marker='o', label='LOO')
        fpr_rf, tpr_rf, _ = roc_curve(res50_truth_building, res50_pred_building_binary)
        plt.plot(fpr_rf[1], tpr_rf[1], label='Res50', marker='o', color=colors[0])
        fpr_rf, tpr_rf, _ = roc_curve(incep_truth_building, incep_pred_building_binary)
        plt.plot(fpr_rf[1], tpr_rf[1], label='Incep', marker='o', color=colors[1])

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Building LOO')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(img_dir, 'prescr_cmp.png'))
        plt.show()
