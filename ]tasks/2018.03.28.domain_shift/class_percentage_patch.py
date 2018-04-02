import os
import imageio
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
import utils
import bohaoCustom.uabPreprocClasses as bPreproc
import uabPreprocClasses
import uab_collectionFunctions
import uab_DataHandlerFunctions
from bohaoCustom import uabMakeNetwork_DeepLabV2

# settings
img_dir, task_dir = utils.get_task_img_folder()

# make network
input_size = [321, 321]
# define place holder
X = tf.placeholder(tf.float32, shape=[None, input_size[0], input_size[1], 3], name='X')
y = tf.placeholder(tf.int32, shape=[None, input_size[0], input_size[1], 1], name='y')
mode = tf.placeholder(tf.bool, name='mode')
model = uabMakeNetwork_DeepLabV2.DeeplabV3({'X':X, 'Y':y},
                                           trainable=mode,
                                           input_size=input_size,)

# create collection
# the original file is in /ei-edl01/data/uab_datasets/inria
blCol = uab_collectionFunctions.uabCollection('inria')
opDetObj = bPreproc.uabOperTileDivide(255)          # inria GT has value 0 and 255, we map it back to 0 and 1
# [3] is the channel id of GT
rescObj = uabPreprocClasses.uabPreprocMultChanOp([], 'GT_Divide.tif', 'Map GT to (0, 1)', [3], opDetObj)
rescObj.run(blCol)
img_mean = blCol.getChannelMeans([0, 1, 2])         # get mean of rgb info

# extract patches
extrObj = uab_DataHandlerFunctions.uabPatchExtr([0, 1, 2, 4], # extract all 4 channels
                                                cSize=input_size, # patch size as 572*572
                                                numPixOverlap=int(model.get_overlap()/2),  # overlap as 92
                                                extSave=['jpg', 'jpg', 'jpg', 'png'], # save rgb files as jpg and gt as png
                                                isTrain=True,
                                                gtInd=3,
                                                pad=model.get_overlap()) # pad around the tiles
patchDir = extrObj.run(blCol)

file_name = os.path.join(patchDir, 'fileList.txt')
with open(file_name, 'r') as f:
    files = f.readlines()
patch_num = len(files)
record_file = os.path.join(task_dir, '{}.txt'.format(os.path.basename(patchDir)))

if not os.path.exists(record_file):
    with open(record_file, 'w+') as f:
        for line in tqdm(files):
            truth_file = line.split(' ')[-1].strip()
            gt = imageio.imread(os.path.join(patchDir, truth_file))
            percentage = np.sum(gt) / input_size[0]**2
            f.write('{} {}\n'.format(truth_file, percentage))

with open(record_file, 'r') as f:
    per_record = f.readlines()
per_nums = np.zeros(patch_num)
for cnt, line in enumerate(per_record):
    per_nums[cnt] = float(line.split(' ')[-1].strip())

# count class patches
class_th = 0.1
city_dict = {'aus':0, 'chi':1, 'kit':2, 'tyr':3, 'vie':4}
city_record = [np.zeros(int(patch_num/5)) for i in range(5)]
city_cnt = np.zeros(5, dtype=np.uint32)
for line in per_record:
    city = line[:3]
    percentage = float(line.split(' ')[-1].strip())
    city_record[city_dict[city]][city_cnt[city_dict[city]]] = percentage
    city_cnt[city_dict[city]] += 1

# make new train set
city_list = ['austin', 'chicago', 'kitsap', 'tyrol-w', 'venna']
for leave_city in city_list:
    file_name = os.path.join(task_dir, 'h0_{}.txt'.format(leave_city))
    with open(file_name, 'w+') as f:
        for cnt, line in tqdm(enumerate(files)):
            if line[:3] == leave_city[:3]:
                if per_nums[cnt] < class_th:
                    f.write(line)
            else:
                f.write(line)
