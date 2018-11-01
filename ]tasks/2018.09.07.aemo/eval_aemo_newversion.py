from nn import unet, nn_utils
from collection import collectionMaker

# settings
class_num = 2
tile_size = (5000, 5000)
suffix = 'aemo_align'
bs = 5
gpu = 1

# define network
patch_size = (572, 572)
unet = unet.UNet(class_num, patch_size, suffix=suffix, batch_size=bs)
overlap = unet.get_overlap()

cm = collectionMaker.read_collection(raw_data_path=r'/home/lab/Documents/bohao/data/aemo/{}'.format(suffix),
                                     field_name='aus10,aus30,aus50',
                                     field_id='',
                                     rgb_ext='.*rgb',
                                     gt_ext='.*gt_d255',
                                     file_ext='tif',
                                     force_run=False,
                                     clc_name=suffix)
cm.print_meta_data()

file_list_train = cm.load_files(field_name='aus10,aus30', field_id='', field_ext='.*rgb,.*gt_d255')
file_list_valid = cm.load_files(field_name='aus50', field_id='', field_ext='.*rgb,.*gt_d255')
chan_mean = cm.meta_data['chan_mean']

nn_utils.tf_warn_level(3)
model_dir = r'/hdd6/Models/aemo/aemo_resize_new_loss/unet_aemo_0_PS(572, 572)_BS5_EP80_LR0.001_DS30_DR0.1'
unet.evaluate(file_list_valid, patch_size, tile_size, bs, chan_mean, model_dir, gpu, save_result_parent_dir='aemo',
              sfn=32, force_run=True, score_results=True, split_char='.', load_epoch_num=4)
