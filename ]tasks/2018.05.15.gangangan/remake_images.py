import os
import imageio
import scipy.misc
import numpy as np
from tqdm import tqdm
import uabCrossValMaker
import bohaoCustom.uabPreprocClasses as bPreproc
import uabPreprocClasses
import uab_collectionFunctions
import uab_DataHandlerFunctions

# settings
output_size = [128, 128]
output_dir = r'/home/lab/Documents/bohao/code/third_party/DCGAN-tensorflow/data/inria'

# create collection
# the original file is in /ei-edl01/data/uab_datasets/inria
blCol = uab_collectionFunctions.uabCollection('inria')
opDetObj = bPreproc.uabOperTileDivide(255)          # inria GT has value 0 and 255, we map it back to 0 and 1
# [3] is the channel id of GT
rescObj = uabPreprocClasses.uabPreprocMultChanOp([], 'GT_Divide.tif', 'Map GT to (0, 1)', [3], opDetObj)
rescObj.run(blCol)
img_mean = np.zeros(3) #blCol.getChannelMeans([0, 1, 2])         # get mean of rgb info

# extract patches
extrObj = uab_DataHandlerFunctions.uabPatchExtr([0, 1, 2, 4], # extract all 4 channels
                                                cSize=(321, 321), # patch size as 572*572
                                                numPixOverlap=0,  # overlap as 92
                                                extSave=['jpg', 'jpg', 'jpg', 'png'], # save rgb files as jpg and gt as png
                                                isTrain=True,
                                                gtInd=3,
                                                pad=0) # pad around the tiles
patchDir = extrObj.run(blCol)

# make data reader
# use uabCrossValMaker to get fileLists for training and validation
idx, file_list = uabCrossValMaker.uabUtilGetFolds(patchDir, 'fileList.txt', 'force_tile')
# use first 5 tiles for validation
file_list_train = uabCrossValMaker.make_file_list_by_key(idx, file_list, [i for i in range(6, 37)])

for files in tqdm(file_list_train):
    img = np.zeros((output_size[0], output_size[1], 3), dtype=np.uint8)
    for cnt, img_file in enumerate(files[:3]):
        img[:, :, cnt] = scipy.misc.imresize(imageio.imread(os.path.join(patchDir, img_file)), output_size)
    imageio.imsave(os.path.join(output_dir, '{}.png'.format(files[0][:-5])), img)
