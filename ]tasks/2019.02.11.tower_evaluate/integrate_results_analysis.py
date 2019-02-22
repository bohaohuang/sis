import os
import scipy.spatial
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import utils
import ersa_utils
import util_functions
from evaluate_utils import local_maxima_suppression
from integrate_conf2 import get_points_between, prune_pairs, connect_lines
from evaluate_tower_performance import get_center_point, read_polygon_csv_data


def add_points(center_points, color='r', size=20, marker='o', alpha=0.5, edgecolor='face'):
    center_points = np.array(center_points)
    plt.scatter(center_points[:, 1], center_points[:, 0], c=color, s=size, marker=marker, alpha=alpha,
                edgecolors=edgecolor)


def get_edge_info(centers, conf_map, radius=1500, width=7, tile_min=(0, 0), tile_max=(5000, 5000)):
    # link towers
    kdt = scipy.spatial.KDTree(np.array(centers))
    linked_pairs = list(kdt.query_pairs(radius))

    # sort by distance
    dist_list = []
    for pair in linked_pairs:
        dist_list.append(np.linalg.norm(center_list[pair[0]] - center_list[pair[1]]))
    sort_idx = np.argsort(dist_list).tolist()
    dist_list = np.sort(dist_list)
    linked_pairs = [linked_pairs[a] for a in sort_idx]

    # compute line confidence
    line_conf = []
    for pair in tqdm(linked_pairs):
        points = get_points_between(centers[pair[0]], centers[pair[1]], width=width, tile_min=tile_min,
                                    tile_max=tile_max)
        conf = 0
        for p in points:
            conf += conf_map[p[0]][p[1]]
        conf /= len(points)
        line_conf.append(conf)

    assert len(linked_pairs) == len(dist_list) == len(line_conf)

    return linked_pairs, dist_list, line_conf


def visualize_with_connected_pairs(raw_rgb, center_list, connected_pairs, style='r', add_fig=False):
    if not add_fig:
        plt.figure(figsize=(12, 8))

    plt.imshow(raw_rgb)
    for pair in connected_pairs:
            plt.plot([center_list[pair[0]][1], center_list[pair[1]][1]],
                     [center_list[pair[0]][0], center_list[pair[1]][0]], style, linewidth=1)

    if not add_fig:
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    # directories
    img_dir, task_dir = utils.get_task_img_folder()
    data_dir = r'/home/lab/Documents/bohao/data/transmission_line'
    raw_dir = os.path.join(data_dir, 'raw')
    conf_dir = r'/media/ei-edl01/user/bh163/tasks/2018.11.16.transmission_line/' \
               r'confmap_uab_UnetCrop_lines_pw30_0_PS(572, 572)_BS5_EP100_LR0.0001_DS60_DR0.1_SFN32'
    lines_dir = r'/media/ei-edl01/data/uab_datasets/lines/data/Original_Tiles'

    city_list = ['AZ_Tucson', 'KS_Colwich_Maize', 'NC_Clyde', 'NC_Wilmington']
    model_name = 'faster_rcnn'

    # settings
    radius = 2000
    width = 7
    th = 5

    for city_id in [3]:
        for tile_id in [2]:
            # load data
            pred_file_name = os.path.join(task_dir, model_name, 'USA_{}_{}.txt'.format(city_list[city_id], tile_id))
            preds = ersa_utils.load_file(pred_file_name)
            csv_file_name = os.path.join(raw_dir, 'USA_{}_{}.csv'.format(city_list[city_id], tile_id))
            raw_rgb = ersa_utils.load_file(os.path.join(raw_dir, 'USA_{}_{}.tif'.format(city_list[city_id], tile_id)))
            conf_img = ersa_utils.load_file(os.path.join(conf_dir, '{}{}.png'.format(city_list[city_id].split('_')[1],
                                                                                     tile_id)))
            line_gt = ersa_utils.load_file(os.path.join(lines_dir, '{}{}_GT.png'.format(city_list[city_id].split('_')[1],
                                                                                        tile_id)))

            # get tower preds
            center_list, conf_list, _ = local_maxima_suppression(preds, 100)

            # get tower truth
            gt_list = []
            csv_file_name = os.path.join(raw_dir, 'USA_{}_{}.csv'.format(city_list[city_id], tile_id))
            for label, bbox in read_polygon_csv_data(csv_file_name):
                y, x = get_center_point(*bbox)
                gt_list.append([y, x])

            # get line confidences
            pairs, dists, confs = get_edge_info(center_list, conf_img, radius=radius, width=width,
                                                tile_min=(0, 0), tile_max=raw_rgb.shape)

            # connect lines
            connected_pairs = connect_lines(pairs, confs, th, cut_n=2)
            connected_pairs, unconnected_pairs = prune_pairs(connected_pairs, center_list)

            # get towers that are not connected
            connected_towers = []
            for p in connected_pairs:
                connected_towers.append(p[0])
                connected_towers.append(p[1])
            connected_towers = list(set(connected_towers))
            unconnected_towers = [a for a in range(len(center_list)) if a not in connected_towers]

            # visualize results
            plt.figure(figsize=(8.5, 8))
            img_with_line = util_functions.add_mask(raw_rgb, line_gt, [0, 255, 0], 1)
            visualize_with_connected_pairs(raw_rgb, center_list, connected_pairs, add_fig=True)
            # visualize_with_connected_pairs(raw_rgb, center_list, unconnected_pairs, style='k', add_fig=True)
            add_points(gt_list, 'b', marker='s', size=80, alpha=1, edgecolor='k')
            add_points([center_list[a] for a in connected_towers], 'r', marker='o', alpha=1, edgecolor='k')
            add_points([center_list[a] for a in unconnected_towers], 'yellow', marker='o', alpha=1, edgecolor='k')
            plt.axis('off')
            plt.tight_layout()
            # plt.savefig(os.path.join(img_dir, '{}_{}_post_result.png'.format(city_list[city_id], tile_id)))
            plt.show()
