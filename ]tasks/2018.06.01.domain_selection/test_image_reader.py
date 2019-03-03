import os
import numpy as np
import matplotlib.pyplot as plt
import sis_utils
import uabCrossValMaker
import uab_collectionFunctions
import uab_DataHandlerFunctions
from bohaoCustom import uabDataReader

img_dir, task_dir = sis_utils.get_task_img_folder()

city_dict = {'austin':0, 'chicago':1, 'kitsap':2, 'tyrol-w':3, 'vienna':4}
city_alpha = [0.5, 0.2, 0.1, 0.1, 0.1]
city_cnt = np.zeros(5)

# create collection
# the original file is in /ei-edl01/data/uab_datasets/inria
blCol = uab_collectionFunctions.uabCollection('inria')
img_mean = blCol.getChannelMeans([0, 1, 2])

# extract patches
extrObj = uab_DataHandlerFunctions.uabPatchExtr([0, 1, 2, 4],  # extract all 4 channels
                                                cSize=(321, 321),  # patch size as 572*572
                                                numPixOverlap=0,  # overlap as 92
                                                extSave=['jpg', 'jpg', 'jpg', 'png'],
                                                # save rgb files as jpg and gt as png
                                                isTrain=True,
                                                gtInd=3,
                                                pad=0)  # pad around the tiles
patchDir = extrObj.run(blCol)

# make data reader
chipFiles = os.path.join(patchDir, 'fileList.txt')
# use uabCrossValMaker to get fileLists for training and validation
idx, file_list = uabCrossValMaker.uabUtilGetFolds(r'/media/ei-edl01/user/bh163/tasks/2018.03.02.res_gan',
                                                  'deeplab_inria_cp_0.txt', 'force_tile')
# use first 5 tiles for validation
file_list_train = uabCrossValMaker.make_file_list_by_key(idx, file_list, [i for i in range(6, 37)])
file_list_valid = uabCrossValMaker.make_file_list_by_key(idx, file_list, [i for i in range(0, 6)])

dataReader_train = uabDataReader.ImageLabelReaderCitySampleControl(
    [3], [0, 1, 2], patchDir, file_list, (321, 321), 5, city_dict, city_alpha, block_mean=np.append([0], img_mean))

for plt_cnt in range(10000):
    _, _, city_batch = dataReader_train.readerAction()
    for city_name in city_batch:
        city_cnt[city_dict[city_name]] += 1

ind = np.arange(5)
plt.bar(ind, city_cnt)
plt.xticks(ind, ['austin', 'chicago', 'kitsap', 'tyrol-w', 'vienna'])
plt.xlabel('City Name')
plt.ylabel('Counts')
plt.savefig(os.path.join(img_dir, 'reader_check.png'))
plt.show()
