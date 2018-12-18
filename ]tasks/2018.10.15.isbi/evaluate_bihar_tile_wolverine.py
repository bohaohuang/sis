import sys
sys.path.append(r'/home/lab/Documents/bohao/code/sis')
sys.path.append(r'/home/lab/Documents/bohao/code/uab')

import os
import time
import imageio
import numpy as np
import tensorflow as tf
from glob import glob
import uabDataReader
import util_functions
import uab_collectionFunctions
from bohaoCustom import uabMakeNetwork_DeepLabV2


def create_fold(fold_path):
    if not os.path.exists(fold_path):
        os.makedirs(fold_path)


def make_preds(model, model_dir, file_list, save_path):
    for file_name in file_list:
        tile_name = os.path.basename(file_name)[:-4]
        print('Evaluating {} ...'.format(tile_name))
        start_time = time.time()
        try:
            tile_size = imageio.imread(file_name).shape[:2]
        except OSError:
            continue
        
        #prepare the reader
        pad = model.get_overlap()
        reader = uabDataReader.ImageLabelReader(gtInds=[0], dataInds=[0], nChannels=3,
                                                parentDir=os.path.dirname(file_name),
                                                chipFiles=[[os.path.basename(file_name)]],
                                                chip_size=input_size, tile_size=tile_size,
                                                batchSize=batch_size, block_mean=img_mean,
                                                overlap=pad, padding=np.array([pad/2, pad/2]),
                                                isTrain=False)
        rManager = reader.readManager

        # run the model
        pred = model.run(pretrained_model_dir=model_dir, test_reader=rManager, tile_size=tile_size,
                         patch_size=input_size, gpu=gpu, load_epoch_num=None, best_model=False)
        print(np.unique(pred), pred.shape)
        
        # save results
        save_name = os.path.join(save_path, '{}_pred.png'.format(tile_name))
        imageio.imsave(save_name, pred.astype(np.uint8))

        duration = time.time() - start_time
        print('duration: {:.2f}'.format(duration))


gpu = 1
batch_size = 1
input_size = [2000, 2000]
util_functions.tf_warn_level(3)
ds_name = 'bihar_building'

# set gpu to use
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISISBLE_DEVICES'] = str(gpu)

# settings
blCol = uab_collectionFunctions.uabCollection(ds_name)
blCol.readMetadata()
img_mean = blCol.getChannelMeans([0, 1, 2])

# make the model
# define place holder
model_dir = r'/media/ei-edl01/user/bh163/models/bihar_building'
X = tf.placeholder(tf.float32, shape=[None, input_size[0], input_size[1], 3], name='X')
y = tf.placeholder(tf.int32, shape=[None, input_size[0], input_size[1], 1], name='y')
mode = tf.placeholder(tf.bool, name='mode')
model = uabMakeNetwork_DeepLabV2.DeeplabV3({'X': X, 'Y': y}, trainable=mode, input_size=input_size, 
                                           batch_size=batch_size, start_filter_num=32)
# create graph
model.create_graph('X', class_num=2)

# walk through folders
data_dir = r'/media/ei-edl01/data/uab_datasets/bihar/patch_for_building_detection'
# fold_ids = ['c', 'd', 'e', 'g', 'h', 'i', 'j', 'k']
fold_ids = ['c']
save_dir = r'/hdd/Sijia/preds/tiles'
for fold_id in fold_ids:
    files = sorted(glob(os.path.join(data_dir, fold_id, '*.tif')))
    save_path = os.path.join(save_dir, fold_id)
    create_fold(save_path)
    make_preds(model, model_dir, files, save_path)
