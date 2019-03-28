"""

"""


# Built-in
import os

# Libs
import pandas as pd
import matplotlib.pyplot as plt

# Own modules
import sis_utils
from rst_utils import misc_utils
from quality_assurance import read_lines_truth
from evaluate_utils import local_maxima_suppression
from make_dataset_demo_figure import read_tower_truth
from post_processing_utils import visualize_with_connected_pairs, add_points

city_list = ['AZ_Tucson', 'KS_Colwich_Maize', 'NC_Clyde', 'NC_Wilmington']


def get_tower_pred(csv_file):
    preds = misc_utils.load_file(csv_file)
    center_list, _, _ = local_maxima_suppression(preds)
    return center_list


def load_data(rgb_dir, task_dir, city_id, tile_id, model_name='faster_rcnn'):
    img = misc_utils.load_file(os.path.join(rgb_dir, 'USA_{}_{}.tif'.format(city_list[city_id], tile_id)))
    pred_vis_tower = misc_utils.load_file(
        os.path.join(task_dir, 'post_{}_{}_{}_pred2.npy'.format(model_name, city_id, tile_id)))
    pred_vis_lines = misc_utils.load_file(
        os.path.join(task_dir, 'post_{}_{}_{}_conn2.npy'.format(model_name, city_id, tile_id)))

    pred_topo_tower = get_tower_pred(os.path.join(task_dir, model_name,
                                                  'USA_{}_{}.txt'.format(city_list[city_id], tile_id)))
    pred_topo_lines = misc_utils.load_file(
        os.path.join(task_dir, '{}_graph_rnn_normal_{}_{}.npy'.format(model_name, city_id, tile_id)))
    pred_combine_tower = pred_vis_tower
    pred_combine_lines = misc_utils.load_file(
        os.path.join(task_dir, '{}_graph_rnn_{}_{}.npy'.format(model_name, city_id, tile_id)))

    tower_file = os.path.join(rgb_dir, 'USA_{}_{}.csv'.format(city_list[city_id], tile_id))
    tower_gt = read_tower_truth(tower_file)
    line_file = os.path.join(rgb_dir, 'USA_{}_{}.csv'.format(city_list[city_id], tile_id))
    line_gt = read_lines_truth(line_file, tower_gt)

    return img, pred_vis_tower, pred_vis_lines, pred_topo_tower, pred_topo_lines, \
           pred_combine_tower, pred_combine_lines, tower_gt, line_gt


def load_data2(rgb_dir, task_dir, city_id, tile_id, model_name='faster_rcnn'):
    img = misc_utils.load_file(os.path.join(rgb_dir, 'USA_{}_{}.tif'.format(city_list[city_id], tile_id)))
    pred_vis_tower = misc_utils.load_file(
        os.path.join(task_dir, 'w3_post_{}_{}_{}_pred2.npy'.format(model_name, city_id, tile_id)))
    pred_vis_lines = misc_utils.load_file(
        os.path.join(task_dir, 'w3_post_{}_{}_{}_conn2.npy'.format(model_name, city_id, tile_id)))

    pred_topo_tower = misc_utils.load_file(
        os.path.join(task_dir, 'w9_post_{}_{}_{}_pred2.npy'.format(model_name, city_id, tile_id)))
    pred_topo_lines = misc_utils.load_file(
        os.path.join(task_dir, 'w9_post_{}_{}_{}_conn2.npy'.format(model_name, city_id, tile_id)))

    pred_combine_tower = misc_utils.load_file(
        os.path.join(task_dir, 'post_{}_{}_{}_pred2.npy'.format(model_name, city_id, tile_id)))
    pred_combine_lines = misc_utils.load_file(
        os.path.join(task_dir, 'post_{}_{}_{}_conn2.npy'.format(model_name, city_id, tile_id)))

    tower_file = os.path.join(rgb_dir, 'USA_{}_{}.csv'.format(city_list[city_id], tile_id))
    tower_gt = read_tower_truth(tower_file)
    line_file = os.path.join(rgb_dir, 'USA_{}_{}.csv'.format(city_list[city_id], tile_id))
    line_gt = read_lines_truth(line_file, tower_gt)

    return img, pred_vis_tower, pred_vis_lines, pred_topo_tower, pred_topo_lines, \
           pred_combine_tower, pred_combine_lines, tower_gt, line_gt


def visualize_results(img, tower, line, alpha=0.5):
    visualize_with_connected_pairs(img, tower, line, add_fig=True)
    add_points(tower, 'b', marker='o', size=40, alpha=alpha, edgecolor='k')
    plt.axis('off')


def make_new_gt(tower_gt, line_gt, city_id, tile_id, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    tower_file_name = os.path.join(save_dir, 'USA_{}_{}_tower.csv'.format(city_list[city_id], tile_id))
    line_file_name = os.path.join(save_dir, 'USA_{}_{}_line.csv'.format(city_list[city_id], tile_id))
    df_tower = pd.DataFrame(tower_gt)
    df_line = pd.DataFrame(line_gt)
    df_tower.to_csv(tower_file_name)
    df_line.to_csv(line_file_name)


if __name__ == '__main__':
    rgb_dir = r'/home/lab/Documents/bohao/data/transmission_line/raw'
    img_dir, task_dir = sis_utils.get_task_img_folder()

    city_id = 3
    tile_id = 3

    img, pred_vis_tower, pred_vis_lines, pred_topo_tower, pred_topo_lines, \
    pred_combine_tower, pred_combine_lines, tower_gt, line_gt = \
        load_data(rgb_dir, task_dir, city_id, tile_id)

    plt.figure(figsize=(18, 4))
    ax1 = plt.subplot(141)
    visualize_results(img, tower_gt, line_gt)
    ax2 = plt.subplot(142, sharex=ax1, sharey=ax1)
    visualize_results(img, pred_vis_tower, pred_vis_lines)
    ax3 = plt.subplot(143, sharex=ax1, sharey=ax1)
    visualize_results(img, pred_topo_tower, pred_topo_lines)
    ax4 = plt.subplot(144, sharex=ax1, sharey=ax1)
    visualize_results(img, pred_combine_tower, pred_combine_lines)
    plt.tight_layout()
    plt.show()

    '''save_dir = os.path.join(task_dir, 'new_gt')
    for city_id in range(4):
        for tile_id in range(1, 4):
            img, pred_vis_tower, pred_vis_lines, pred_topo_tower, pred_topo_lines, \
            pred_combine_tower, pred_combine_lines, tower_gt, line_gt = \
                load_data(rgb_dir, task_dir, city_id, tile_id)

            make_new_gt(tower_gt, line_gt, city_id, tile_id, save_dir)'''
