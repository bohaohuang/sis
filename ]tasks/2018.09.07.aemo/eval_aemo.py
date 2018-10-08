import numpy as np
from nn import unet, nn_utils
from collection import collectionMaker, collectionEditor

# settings
class_num = 2
patch_size = (572, 572)
tile_size = (5000, 5000)
suffix = 'aemo'
lr = 1e-5
ds = 20
dr = 0.1
epochs = 10
bs = 5
gpu = 1

# define network
unet = unet.UNet(class_num, patch_size, suffix=suffix, learn_rate=lr, decay_step=ds, decay_rate=dr,
                 epochs=epochs, batch_size=bs)
overlap = unet.get_overlap()

cm = collectionMaker.read_collection(raw_data_path=r'/home/lab/Documents/bohao/data/aemo',
                                     field_name='aus',
                                     field_id='',
                                     rgb_ext='.*rgb',
                                     gt_ext='.*gt',
                                     file_ext='tif',
                                     force_run=False,
                                     clc_name='aemo')
gt_d255 = collectionEditor.SingleChanMult(cm.clc_dir, 1/255, ['.*gt', 'gt_d255']).\
    run(force_run=False, file_ext='tif', d_type=np.uint8,)
cm.replace_channel(gt_d255.files, True, ['gt', 'gt_d255'])
cm.print_meta_data()

file_list_valid = cm.load_files(field_id='', field_ext='.*rgb,.*gt_d255')
chan_mean = cm.meta_data['chan_mean'][:3]

nn_utils.tf_warn_level(3)
model_dir = r'/hdd6/Models/aemo/unet_aemo_PS(572, 572)_BS5_EP60_LR0.001_DS40_DR0.1'
unet.evaluate(file_list_valid, patch_size, tile_size, bs, chan_mean, model_dir, gpu, save_result_parent_dir='aemo',
              sfn=32, force_run=True, score_results=True)
