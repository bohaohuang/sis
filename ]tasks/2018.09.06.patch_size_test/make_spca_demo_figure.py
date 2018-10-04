import os
import ersa_utils
from visualize import visualize_utils

data_dir = r'/media/ei-edl01/data/uab_datasets/spca/data/Original_Tiles'
img_name = 'Fresno456'

rgb_file = os.path.join(data_dir, '{}_RGB.jpg'.format(img_name))
gt_file = os.path.join(data_dir, '{}_GT.png'.format(img_name))
rgb = ersa_utils.load_file(rgb_file)
gt = ersa_utils.load_file(gt_file)

visualize_utils.compare_two_figure(rgb, gt)
