import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from natsort import natsorted
import utils
import ersa_utils
from visualize import visualize_utils


def clean_conf_map(pred):
    kernel = np.ones((25, 25), np.uint8)
    pred = cv2.morphologyEx(pred, cv2.MORPH_OPEN, np.ones((10, 10), np.uint8))
    visualize_utils.compare_figures([rgb, gt, conf, pred], (2, 2), fig_size=(10, 8), show_axis=True)
    pred = cv2.morphologyEx(pred, cv2.MORPH_CLOSE, kernel, iterations=2)
    return pred

# settings
img_dir, task_dir = utils.get_task_img_folder()
ds_name = 'lines'
weight = 50
city_list = ['Tucson', 'Colwich', 'Clyde', 'Wilmington']

for city_idx in range(3, 4):#range(len(city_list)):
    model_name = 'UnetCrop_lines_city{}_pw{}_0_PS(572, 572)_BS5_EP50_LR0.0001_DS30_DR0.1_SFN32'.format(city_idx, weight)
    results_dir = os.path.join(r'/hdd/Results', ds_name, model_name, ds_name+'_tw1', 'pred')
    conf_dir = os.path.join(task_dir, 'confmap_uab_{}'.format(model_name))
    raw_data_dir = r'/home/lab/Documents/bohao/data/transmission_line/raw'
    rgb_files = natsorted([a for a in glob(os.path.join(raw_data_dir, '*.tif'))
                           if 'multiclass' not in a])
    '''gt_files = natsorted([a for a in glob(os.path.join(raw_data_dir, '*.tif'))
                          if 'multiclass' in a])'''
    gt_files = natsorted(glob(os.path.join(r'/media/ei-edl01/data/uab_datasets/lines_tw1/data/Original_Tiles',
                                           '*_GT.png')))
    pred_files = natsorted(glob(os.path.join(conf_dir, '*.png')))

    for conf_file in pred_files:
        city_with_id = os.path.basename(conf_file)[:-4]
        city_name = ''.join([a for a in city_with_id if not a.isdigit()])
        city_id = ''.join([a for a in city_with_id if a.isdigit()])

        if city_name == city_list[city_idx]:
            rgb_file = [a for a in rgb_files if city_name in a and city_id in a][0]
            gt_file = [a for a in gt_files if city_name in a and city_id in a][0]
            pred_file = os.path.join(results_dir, '{}{}.png'.format(city_name, city_id))

            rgb = ersa_utils.load_file(rgb_file)
            gt = ersa_utils.load_file(gt_file)
            conf = ersa_utils.load_file(conf_file)
            pred = ersa_utils.load_file(pred_file)

            visualize_utils.compare_figures([rgb, gt, conf, pred], (2, 2), fig_size=(10, 8), show_axis=True)
