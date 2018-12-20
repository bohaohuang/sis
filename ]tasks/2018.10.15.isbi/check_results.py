import os
from glob import glob
import ersa_utils
from visualize import visualize_utils

city_id = 'c'
pred_dir = r'/hdd/Sijia/preds/tiles/{}'.format(city_id)
rgb_dir = r'/media/ei-edl01/data/uab_datasets/bihar/patch_for_building_detection/{}'.format(city_id)
pred_files = sorted(glob(os.path.join(pred_dir, '*.png')))

for pred_file in pred_files:
    suffix = os.path.basename(pred_file)[:-9]
    rgb_file = os.path.join(rgb_dir, '{}.tif'.format(suffix))

    pred = ersa_utils.load_file(pred_file)
    rgb = ersa_utils.load_file(rgb_file)

    visualize_utils.compare_figures([rgb, pred], (1, 2), fig_size=(12, 5))
