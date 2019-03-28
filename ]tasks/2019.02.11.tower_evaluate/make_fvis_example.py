"""

"""


# Built-in
import os

# Libs
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter

# Own modules
import sis_utils
import util_functions
from rst_utils import misc_utils
from visualize import visualize_utils
from quality_assurance import read_lines_truth
from make_dataset_demo_figure import read_tower_truth


city_list = ['AZ_Tucson', 'KS_Colwich_Maize', 'NC_Clyde', 'NC_Wilmington']


def load_data(rgb_dir, conf_dir):
    img = misc_utils.load_file(os.path.join(rgb_dir, 'USA_{}_{}.tif'.format(city_list[city_id], tile_id)))
    tower_file = os.path.join(rgb_dir, 'USA_{}_{}.csv'.format(city_list[city_id], tile_id))
    tower_gt = read_tower_truth(tower_file)
    line_file = os.path.join(rgb_dir, 'USA_{}_{}.csv'.format(city_list[city_id], tile_id))
    line_gt = read_lines_truth(line_file, tower_gt)
    line_conf = misc_utils.load_file(
        os.path.join(conf_dir, '{}{}.png'.format(city_list[city_id].split('_')[1], tile_id)))

    return img, tower_gt, line_gt, line_conf


def visualize_data(img, line_conf, img_dir):
    line_conf = gaussian_filter(line_conf, 25)
    img = img[7110:7500, 6880:7420, :]
    line_conf = line_conf[7110:7500, 6880:7420]
    visualize_utils.compare_figures((img, line_conf), (1, 2), show_axis=False, fig_size=(24, 10), show_fig=False)

    save_dir = os.path.join(img_dir, 'demo_save')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.savefig(os.path.join(save_dir, 'demo.png'))
    #misc_utils.save_file(os.path.join(save_dir, 'orig.png'), img)
    #misc_utils.save_file(os.path.join(save_dir, 'conf.png'), line_conf)


if __name__ == '__main__':
    rgb_dir = r'/home/lab/Documents/bohao/data/transmission_line/raw'
    conf_dir = r'/media/ei-edl01/user/bh163/tasks/2018.11.16.transmission_line/' \
               r'confmap_uab_UnetCrop_lines_pw30_0_PS(572, 572)_BS5_EP100_LR0.0001_DS60_DR0.1_SFN32'
    img_dir, task_dir = sis_utils.get_task_img_folder()

    city_id = 2
    tile_id = 3

    img, tower_gt, line_gt, line_conf = load_data(rgb_dir, conf_dir)
    visualize_data(img, line_conf, img_dir)
