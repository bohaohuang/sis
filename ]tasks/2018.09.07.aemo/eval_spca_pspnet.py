import tensorflow as tf
from nn import pspnet, nn_utils
from collection import collectionMaker

# settings
class_num = 2
patch_size = (384, 384)
tile_size = (5000, 5000)
suffix = 'spca'
batch_size = 5
data_dir = r'/media/ei-edl01/data/uab_datasets/spca/data/Original_Tiles'
ds_name = 'spca'
gpu = 1

model_dirs = [#r'/hdd6/Models/spca/psp101/pspnet_spca_PS(384, 384)_BS5_EP100_LR0.0001_DS40_DR0.1',
              #r'/hdd6/Models/spca/psp101/pspnet_spca_PS(384, 384)_BS5_EP100_LR0.0001_DS100_DR0.1',
              r'/hdd6/Models/spca/psp101/pspnet_spca_PS(384, 384)_BS5_EP100_LR0.001_DS40_DR0.1']

# define network
for model_dir in model_dirs:
    tf.reset_default_graph()
    model = pspnet.PSPNet(class_num, patch_size, suffix=suffix,  batch_size=batch_size)
    overlap = model.get_overlap()

    cm = collectionMaker.read_collection(raw_data_path=data_dir,
                                         field_name='Fresno,Modesto,Stockton',
                                         field_id=','.join(str(i) for i in range(663)),
                                         rgb_ext='RGB',
                                         gt_ext='GT',
                                         file_ext='jpg,png',
                                         force_run=False,
                                         clc_name=ds_name)
    cm.print_meta_data()
    file_list_train = cm.load_files(field_id=','.join(str(i) for i in range(0, 250)), field_ext='RGB,GT')
    file_list_valid = cm.load_files(field_id=','.join(str(i) for i in range(250, 500)), field_ext='RGB,GT')
    chan_mean = cm.meta_data['chan_mean'][:3]

    nn_utils.tf_warn_level(3)
    model.evaluate(file_list_valid, patch_size, tile_size, batch_size, chan_mean, model_dir, gpu,
                   save_result_parent_dir='spca', sfn=32, force_run=True, score_results=True, split_char='.')
