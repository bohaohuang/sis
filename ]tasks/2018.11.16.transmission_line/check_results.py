import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from natsort import natsorted
import utils
import ersa_utils
from visualize import visualize_utils

# settings
img_dir, task_dir = utils.get_task_img_folder()
ds_name = 'lines'
model_name = 'UnetCrop_lines_pw1_0_PS(572, 572)_BS5_EP100_LR0.0001_DS60_DR0.1_SFN32'
results_dir = os.path.join(r'/hdd/Results', ds_name, model_name, ds_name)
conf_dir = os.path.join(task_dir, 'confmap_uab_{}'.format(model_name))
raw_data_dir = r'/home/lab/Documents/bohao/data/transmission_line/raw'
rgb_files = natsorted([a for a in glob(os.path.join(raw_data_dir, '*.tif'))
                       if 'multiclass' not in a])
gt_files = natsorted([a for a in glob(os.path.join(raw_data_dir, '*.tif'))
                      if 'multiclass' in a])

pred_files = natsorted(glob(os.path.join(conf_dir, '*.npy')))
for conf_file in pred_files:
    city_with_id = os.path.basename(conf_file)[:-4]
    city_name = ''.join([a for a in city_with_id if not a.isdigit()])
    city_id = ''.join([a for a in city_with_id if a.isdigit()])

    rgb_file = [a for a in rgb_files if city_name in a and city_id in a][0]
    gt_file = [a for a in gt_files if city_name in a and city_id in a][0]

    rgb = ersa_utils.load_file(rgb_file)
    gt = ersa_utils.load_file(gt_file)
    conf = ersa_utils.load_file(conf_file)

    visualize_utils.compare_three_figure(rgb, gt, conf)
