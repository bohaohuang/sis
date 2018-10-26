import os
import numpy as np
import tensorflow as tf
import ersa_utils
from nn import unet, nn_utils
from collection import collectionMaker

# settings
class_num = 2
tile_size = (5000, 5000)
suffix = 'aemo_hist'
bs = 5
gpu = 0
temp_name = r'/work/bh163/misc/aemo_temp'

cm = collectionMaker.read_collection(raw_data_path=r'/work/bh163/data/aemo_hist',
                                     field_name='aus10,aus30,aus50',
                                     field_id='',
                                     rgb_ext='.*rgb',
                                     gt_ext='.*gt',
                                     file_ext='tif',
                                     force_run=False,
                                     clc_name=suffix)
cm.print_meta_data()

file_list_train = cm.load_files(field_name='aus10,aus30', field_id='', field_ext='.*rgb,.*gt')
file_list_valid = cm.load_files(field_name='aus50', field_id='', field_ext='.*rgb,.*gt')
chan_mean = cm.meta_data['chan_mean']

nn_utils.tf_warn_level(3)
model_dir = r'/work/bh163/misc/unet_reweight'

fold_num = '1'
beta_vals = np.arange(0.5, 1.1, 0.1)
alpha_vals = np.arange(-20, 21, 1)
gamma_vals = np.arange(0.5, 1.6, 0.1)

record_file = os.path.join(temp_name, 'cust_func_record.txt')

for beta in beta_vals:
    for alpha in alpha_vals:
        for gamma in gamma_vals:
            tf.reset_default_graph()
            # define network
            patch_size = (572, 572)
            model = unet.UNet(class_num, patch_size, suffix=suffix, batch_size=bs)
            overlap = model.get_overlap()

            file_list_valid_new = []
            for file in file_list_valid:
                # make temp rgb image
                rgb = ersa_utils.load_file(file[0])
                rgb = beta * (rgb - alpha - chan_mean) ** gamma
                save_file = os.path.join(temp_name, fold_num, os.path.basename(file[0]))
                ersa_utils.save_file(save_file, rgb)
                file_list_valid_new.append([save_file, file[1]])
            chan_mean_new = beta * (chan_mean - alpha - chan_mean) ** gamma

            print('Evaluating beta={}, alpha={}, gamma={}...'.format(beta, alpha, gamma))
            _, _, iou = model.evaluate(file_list_valid, patch_size, tile_size, bs, chan_mean, model_dir, gpu,
                                       save_result_parent_dir='aemo', sfn=32, force_run=True, score_results=True,
                                       split_char='.')

            with open(record_file, 'a') as f:
                f.write('{} {} {} {}\n'.format(beta, alpha, gamma, iou))
