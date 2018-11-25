import os
from glob import glob
import ersa_utils
from visualize import visualize_utils
from collection import collectionMaker

# settings
ds_name = 'infrastructure'
pred_dir = r'/hdd/Results/infrastructure/unet_5objs_weight0.999_PS(572, 572)_BS5_EP100_LR0.0001_DS60_DR0.1/default/pred'
pred_files = sorted(glob(os.path.join(pred_dir, '*.png')))

# define network
cm = collectionMaker.read_collection(clc_name=ds_name)
cm.print_meta_data()
file_list_valid = cm.load_files(field_name='Tucson,Colwich,Clyde,Wilmington', field_id='1,2,3',
                                field_ext='RGB,GT_switch')
chan_mean = cm.meta_data['chan_mean']

for (rgb_name, gt_name), pred_name in zip(file_list_valid, pred_files):
    rgb, gt, pred = ersa_utils.load_file(rgb_name), ersa_utils.load_file(gt_name), ersa_utils.load_file(pred_name)
    visualize_utils.compare_three_figure(rgb, gt, pred)
