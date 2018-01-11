import os
import imageio
import numpy as np
import uab_collectionFunctions
import uabPreprocClasses
import uab_DataHandlerFunctions
from bohaoCustom import uabPreprocClasses as bPreproc


# create collection
# the original file is in /ei-edl01/data/uab_datasets/inria
blCol = uab_collectionFunctions.uabCollection('inria')
opDetObj = bPreproc.uabOperTileDivide(255)          # inria GT has value 0 and 255, we map it back to 0 and 1
# [3] is the channel id of GT
rescObj = uabPreprocClasses.uabPreprocMultChanOp([], 'GT_Divide.tif', 'Map GT to (0, 1)', [3], opDetObj)
rescObj.run(blCol)
img_mean = blCol.getChannelMeans([0, 1, 2])         # get mean of rgb info
print(blCol.readMetadata())                         # now inria collection has 4 channels, the last one is GT with (0, 1)

# extract patches
chip_size = (844, 844)
extrObj = uab_DataHandlerFunctions.uabPatchExtr([0, 1, 2, 4], # extract all 4 channels
                                                cSize=chip_size, # patch size as 572*572
                                                numPixOverlap=92,  # overlap as 92
                                                extSave=['jpg', 'jpg', 'jpg', 'png'], # save rgb files as jpg and gt as png
                                                isTrain=True,
                                                gtInd=3,
                                                pad=184) # pad around the tiles
patchDir = extrObj.run(blCol)

# make data reader
chipFiles = os.path.join(patchDir, 'fileList.txt')
with open(chipFiles, 'r') as f:
    cFiles = f.readlines()

# extract patches
idx = np.random.permutation(len(cFiles))
parent_dir = patchDir
child_dir = r'/hdd/Temp/Inria_289_184'
# rename data
for i in idx[:10]:
    files = cFiles[i]
    files = files.strip().split(' ')
    print(files)

    for cnt, file in enumerate(files):
        img = imageio.imread(os.path.join(parent_dir, file))
        img = img[92:844-92, 92:844-92]
        print(img.shape)
