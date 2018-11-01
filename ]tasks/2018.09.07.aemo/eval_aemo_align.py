import os
import scipy.misc
import numpy as np
from glob import glob
from tqdm import tqdm
import ersa_utils
import processBlock
from nn import unet, deeplab, nn_utils
from preprocess import histMatching
from collection import collectionMaker, collectionEditor

# settings
class_num = 2
tile_size = (5000, 5000)
clc_name = 'aemo_align'
bs = 5
gpu = 0

# define network
patch_size = (572, 572)
unet = unet.UNet(class_num, patch_size, batch_size=bs)
overlap = unet.get_overlap()

cm = collectionMaker.read_collection(raw_data_path=r'/home/lab/Documents/bohao/data/aemo/{}'.format(clc_name),
                                     field_name='aus10,aus30,aus50',
                                     field_id='',
                                     rgb_ext='.*rgb',
                                     gt_ext='.*gt',
                                     file_ext='tif',
                                     force_run=False,
                                     clc_name=clc_name)
cm.print_meta_data()

file_list_train = cm.load_files(field_name='aus10,aus30', field_id='', field_ext='.*rgb,.*gt_d255')
file_list_valid = cm.load_files(field_name='aus50', field_id='', field_ext='.*rgb,.*gt_d255')
chan_mean = cm.meta_data['chan_mean']

nn_utils.tf_warn_level(3)
model_dir = r'/hdd6/Models/aemo/aemo_resize/unet_aemo_0_PS(572, 572)_BS5_EP80_LR0.001_DS30_DR0.1'
unet.evaluate(file_list_valid, patch_size, tile_size, bs, chan_mean, model_dir, gpu, save_result_parent_dir='aemo',
              sfn=32, force_run=True, score_results=True, split_char='.', load_epoch_num=4)
