import os
import csv
import imageio
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import utils
import util_functions
import uabCrossValMaker
import uabPreprocClasses
import uab_collectionFunctions
import uab_DataHandlerFunctions
import bohaoCustom.uabPreprocClasses as bPreproc
from bohaoCustom import uabMakeNetwork_UNetEncoder


def patch_iterator(parent_dir, file_list, patch_size, img_mean):
    for files in file_list:
        img = np.zeros((patch_size[0], patch_size[1], 3), dtype=np.uint8)
        for cnt, f in enumerate(files[:3]):
            img[:, :, cnt] = imageio.imread(os.path.join(parent_dir, f))
        yield np.expand_dims(img - img_mean, axis=0)


# settings
gpu = 1
batch_size = 5
input_size = [256, 256]
latent_num = 500
img_dir, task_dir = utils.get_task_img_folder()
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

# make the model
# define place holder
X = tf.placeholder(tf.float32, shape=[None, input_size[0], input_size[1], 3], name='X')
y = tf.placeholder(tf.int32, shape=[None, input_size[0], input_size[1], 1], name='y')
mode = tf.placeholder(tf.bool, name='mode')
model = uabMakeNetwork_UNetEncoder.VGGVAE({'X':X, 'Y':y},
                                          trainable=mode,
                                          input_size=input_size,
                                          batch_size=1,
                                          start_filter_num=32,
                                          latent_num=latent_num)
# create graph
model.create_graph('X', class_num=3)

model_dir = r'/hdd6/Models/VGGVAE/VGGVAE_inria_z500_0_PS(256, 256)_BS5_EP400_LR1e-05_DS200.0_DR0.5_SFN32'
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
# make data reader
# use uabCrossValMaker to get fileLists for training and validation
idx, file_list = uabCrossValMaker.uabUtilGetFolds(patchDir, 'fileList.txt', 'force_tile')

# make reader
# prepare the reader
reader = patch_iterator(patchDir, file_list, input_size, img_mean)
file_name = os.path.join(task_dir, '{}_inria.csv'.format(model_dir.split('/')[-1]))
patch_file_name = os.path.join(task_dir, '{}_inria.txt'.format(model_dir.split('/')[-1]))

# evaluate on tiles
with open(file_name, 'w+') as f:
    with open(patch_file_name, 'w+') as f2:
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            model.load(model_dir, sess, best_model=False)
            encode_reader = model.encoding('X', sess, reader)

            for file in tqdm(file_list):
                patch_name = file[0].split('.')[0][:-5]
                encoded = next(encode_reader)
                writer = csv.writer(f, lineterminator='\n')
                writer.writerow(['{}'.format(x) for x in encoded])
                f2.write('{}\n'.format(patch_name))
