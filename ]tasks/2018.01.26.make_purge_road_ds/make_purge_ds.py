import os
import imageio
import numpy as np
import bohaoCustom.uabPreprocClasses as bPreproc
import uabPreprocClasses
import uab_collectionFunctions
import uab_DataHandlerFunctions
from shutil import copyfile
from tqdm import tqdm


def compute_missing_percentage(rgb):
    stack = np.sum(rgb, axis=2)

    def white_pixel(a):
        if a == 255*3:
            return 1
        else:
            return 0

    map_func = np.vectorize(white_pixel)
    mpixel_map = map_func(stack)
    return np.sum(mpixel_map)/(mpixel_map.shape[0] ** 2), 1-mpixel_map


blCol = uab_collectionFunctions.uabCollection('road')
opDetObj = bPreproc.uabOperTileDivide(255)          # inria GT has value 0 and 255, we map it back to 0 and 1
# [0] is the channel id of GT
rescObj = uabPreprocClasses.uabPreprocMultChanOp([], 'GT_Divide.tif', 'Map GT to (0, 1)', [0], opDetObj)
rescObj.run(blCol)
blCol.readMetadata()
img_mean = blCol.getChannelMeans([1, 2, 3])         # get mean of rgb info

# extract patches
extrObj = uab_DataHandlerFunctions.uabPatchExtr([1, 2, 3, 4], # extract all 4 channels
                                                cSize=(572, 572), # patch size as 572*572
                                                numPixOverlap=46,  # half overlap for this
                                                extSave=['jpg', 'jpg', 'jpg', 'png'], # save rgb files as jpg and gt as png
                                                isTrain=True,
                                                gtInd=3,
                                                pad=184) # pad around the tiles
patchDir = extrObj.run(blCol)
patchDir2 = r'/hdd/uab_datasets/Results/PatchExtr/road/chipExtrRegPurge_cSz572x572_pad184'
if not os.path.exists(patchDir2):
    os.makedirs(patchDir2)


files = os.path.join(patchDir, 'fileList.txt')
with open(files, 'r') as f:
    file_list = f.readlines()
file_list_new = []

for file in tqdm(file_list):
    f_array = file.strip().split(' ')
    rgb = []
    for i in f_array[:3]:
        rgb.append(imageio.imread(os.path.join(patchDir, i)))
    rgb = np.dstack(rgb)
    gt = imageio.imread(os.path.join(patchDir, f_array[-1]))
    m_pcent, mask = compute_missing_percentage(rgb)

    if m_pcent < 0.2:
        file_list_new.append(file)
        for i in f_array[:3]:
            copyfile(os.path.join(patchDir, i), os.path.join(patchDir2, i))
        if m_pcent > 0:
            gt_new = (gt * mask).astype(np.uint8)
            imageio.imsave(os.path.join(patchDir2, f_array[-1]), gt_new)
        else:
            copyfile(os.path.join(patchDir, f_array[-1]), os.path.join(patchDir2, f_array[-1]))

files = os.path.join(patchDir2, 'fileList.txt')
with open(files, 'w+') as f:
    for file in file_list_new:
        f.write(file)
