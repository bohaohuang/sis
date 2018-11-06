import uab_collectionFunctions
from nn import unet, nn_utils
from collection import collectionMaker

class_num = 2
patch_size = (572, 572)
tile_size = (5000, 5000)
batch_size = 1
gpu = 0
model_dir = r'/hdd6/Models/Inria_decay/UnetCrop_inria_decay_0_PS(572, 572)_BS5_EP100_LR0.0001_DS60.0_DR0.1_SFN32'

cm = collectionMaker.read_collection('inria')
cm.print_meta_data()

file_list_valid = cm.load_files(field_id=','.join(str(i) for i in range(5)), field_ext='RGB,gt_d255')
chan_mean = cm.meta_data['chan_mean']

blCol = uab_collectionFunctions.uabCollection('inria')
blCol.readMetadata()
#chan_mean = blCol.getChannelMeans([0, 1, 2])

nn_utils.tf_warn_level(3)
model = unet.UNet(class_num, patch_size)

model.evaluate(file_list_valid, patch_size, tile_size, batch_size, chan_mean, model_dir, gpu,
               save_result_parent_dir='spca', sfn=32, force_run=True, score_results=True, split_char='.',
               load_epoch_num=95)
