import os
import keras
import numpy as np
import utils
from tqdm import tqdm
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
        # idx_batch = np.random.permutation(len(chipFiles))
        idx_batch = np.arange(len(chipFiles))
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
    batch_size = 132
    prescr_name = 'res50'
    blCol = uab_collectionFunctions.uabCollection('inria')
    img_mean = blCol.getChannelMeans([0, 1, 2])
    img_dir, task_dir = utils.get_task_img_folder()

    if prescr_name == 'incep':
        center_crop = (299, 299)
    elif prescr_name == 'res50':
        center_crop = (224, 224)
    else:
        raise KeyError

    for city_num in tqdm(range(5)):

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
        file_list_valid = uabCrossValMaker.make_file_list_by_key(idx, file_list, [city_num])

        dataReader_valid = ImageLabelReaderBuilding(
            [3], [0, 1, 2], patchDir, file_list_valid, patch_size, batch_size, center_crop, 0.1,
            block_mean=np.append([0], img_mean)).readManager

        pred_building_binary = np.zeros(len(file_list_valid))
        for cnt, (img, gt) in enumerate(dataReader_valid):
            pred = model.predict_on_batch(img)
            pred_building_binary[cnt*batch_size:(cnt+1)*batch_size] = np.argmax(pred, axis=1)

        # pred_building_binary = [item for sublist in pred_building_binary for item in sublist]
        np.save(os.path.join(task_dir, '{}_pred_building_binary_{}.npy'.
                             format(prescr_name, city_num)), pred_building_binary)
