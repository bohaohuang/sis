import os
import imageio
from glob import glob
from shutil import copyfile

'''origin_dir = r'/media/ei-edl01/data/remote_sensing_data/urban_mapper/trOut'
new_dir = r'/media/ei-edl01/data/uab_datasets/um/data/Original_Tiles'

rgb_files = sorted(glob(os.path.join(origin_dir, '*pRGB.tif')))
gt_files = sorted(glob(os.path.join(origin_dir, '*pcustGT.png')))

for (rgb, gt) in zip(rgb_files, gt_files):
    rgb_file_name = rgb.split('/')[-1]
    gt_file_name = gt.split('/')[-1]
    city_name = rgb_file_name.split('_')[0]
    tile_num = int(rgb_file_name.split('_')[2])

    rgb_file_name_new = '{}{}_RGB.tif'.format(city_name, tile_num)
    gt_file_name_new = '{}{}_GT.png'.format(city_name, tile_num)

    copyfile(rgb, os.path.join(new_dir, rgb_file_name_new))
    copyfile(gt, os.path.join(new_dir, gt_file_name_new))

    print('{} {} done'.format(city_name, tile_num))

gt_file = os.path.join(new_dir, 'JAX4_GT.png')
image = imageio.imread(gt_file)
import matplotlib.pyplot as plt
plt.hist(image.flatten())
plt.show()'''

import uab_collectionFunctions
import uabPreprocClasses
import bohaoCustom.uabPreprocClasses as bPreproc

blCol = uab_collectionFunctions.uabCollection('um')
print(blCol.readMetadata())
blCol.getMetaDataInfo([0, 1, 2])
print(blCol.getChannelMeans([0, 1, 2]))

opDetObj = bPreproc.uabOperTileDivide(255)          # inria GT has value 0 and 255, we map it back to 0 and 1
# [3] is the channel id of GT
rescObj = uabPreprocClasses.uabPreprocMultChanOp([], 'GT_Divide.tif', 'Map GT to (0, 1)', [3], opDetObj)
rescObj.run(blCol)
print(blCol.readMetadata())

# meta = blCol.getMetaDataInfo([0, 1, 2], forcerun=True)
# print(meta)

blCol = uab_collectionFunctions.uabCollection('inria')
meta = blCol.getMetaDataInfo([0, 1, 2], forcerun=True)
print(meta)
