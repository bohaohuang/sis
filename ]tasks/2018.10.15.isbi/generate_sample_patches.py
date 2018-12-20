import os
import matplotlib.pyplot as plt
from glob import glob
from natsort import natsorted
import utils
import ersa_utils
from visualize import visualize_utils

img_dir, task_dir = utils.get_task_img_folder()

data_dir = r'/media/ei-edl01/data/uab_datasets/bihar_building/data/Original_Tiles'
rgb_files = natsorted(glob(os.path.join(data_dir, '*.tif')))
gt_files = natsorted(glob(os.path.join(data_dir, '*.png')))
save_dir = os.path.join(img_dir, 'annotation_cmp')
ersa_utils.make_dir_if_not_exist(save_dir)

for cnt, (rgb_file, gt_file) in enumerate(zip(rgb_files, gt_files)):
    rgb = ersa_utils.load_file(rgb_file)
    gt = ersa_utils.load_file(gt_file)
    visualize_utils.compare_figures([rgb, gt], (1, 2), fig_size=(12, 5), show_fig=False)
    plt.savefig(os.path.join(save_dir, '{}.png'.format(cnt)))
    plt.close()
