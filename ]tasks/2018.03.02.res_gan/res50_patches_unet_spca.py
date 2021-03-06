import os
import csv
import keras
import imageio
import scipy.misc
import numpy as np
import tensorflow as tf
import uab_collectionFunctions
import uab_DataHandlerFunctions
from tqdm import tqdm
import sis_utils
from bohaoCustom import uabMakeNetwork_UNet

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
img_dir, task_dir = sis_utils.get_task_img_folder()

# make network
input_size = (572, 572)
input_size_fit = (224, 224)
# define place holder
X = tf.placeholder(tf.float32, shape=[None, input_size[0], input_size[1], 3], name='X')
y = tf.placeholder(tf.int32, shape=[None, input_size[0], input_size[1], 1], name='y')
mode = tf.placeholder(tf.bool, name='mode')
model = uabMakeNetwork_UNet.UnetModelCrop({'X':X, 'Y':y},
                                          trainable=mode,
                                          input_size=input_size,)

# create collection
# the original file is in /ei-edl01/data/uab_datasets/inria
blCol = uab_collectionFunctions.uabCollection('spca')
img_mean = blCol.getChannelMeans([1, 2, 3])         # get mean of rgb info

# extract patches
extrObj = uab_DataHandlerFunctions.uabPatchExtr([0, 1, 2, 3], # extract all 4 channels
                                                cSize=input_size, # patch size as 572*572
                                                numPixOverlap=int(model.get_overlap()/2),  # overlap as 92
                                                extSave=['png', 'jpg', 'jpg', 'jpg'], # save rgb files as jpg and gt as png
                                                isTrain=True,
                                                gtInd=0,
                                                pad=model.get_overlap()) # pad around the tiles
patchDir = extrObj.run(blCol)

file_name = os.path.join(patchDir, 'fileList.txt')
with open(file_name, 'r') as f:
    files = f.readlines()

res50 = keras.applications.resnet50.ResNet50(include_top=True, weights='imagenet')
file_name = os.path.join(task_dir, 'temp', 'res50_fc1000_sp_unet.csv')
patch_file_name = os.path.join(task_dir, 'temp', 'res50_fc1000_sp_unet.txt')
print(file_name)
print(patch_file_name)
with open(file_name, 'w+') as f:
    with open(patch_file_name, 'w+') as f2:
        for file_line in tqdm(files):
            patch_name = file_line.split('.')[0][:-5]
            img = np.zeros((input_size[0], input_size[1], 3), dtype=np.uint8)
            for cnt, file in enumerate(file_line.strip().split(' ')[:3]):
                img[:, :, cnt] = imageio.imread(os.path.join(patchDir, file))
            img = np.expand_dims(scipy.misc.imresize(img, input_size_fit), axis=0)

            fc1000 = res50.predict(img).reshape((-1,)).tolist()
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow(['{}'.format(x) for x in fc1000])
            f2.write('{}\n'.format(patch_name))
