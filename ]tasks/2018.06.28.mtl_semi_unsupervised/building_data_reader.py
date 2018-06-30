import os
import numpy as np
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
        while True:
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
