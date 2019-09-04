"""

"""


# Built-in
import os
import numpy as np
from glob import glob

# Libs
import cv2
import matplotlib.pyplot as plt
from natsort import natsorted

# Own modules
import sis_utils
import util_functions
from rst_utils import misc_utils
from evaluate_tower_performance import link_pred_gt
from line_length_stats import read_line_csv_data, add_point_if_not_nearby
from post_processing_utils import read_polygon_csv_data, get_center_point, order_pair, visualize_with_connected_pairs


def grid_score(tower_gt, tower_pred, line_gt, line_pred, link_list, score_type='all'):
    assert score_type in ['Graph', 'Tower', 'Line']

    cnt_obj = 0
    for a in link_list:
        if a > -1:
            cnt_obj += 1

    lp = []
    for cp in line_pred:
        lp.append(order_pair(*cp))
    lp = list(set(lp))

    cnt_pred = 0
    for cp in lp:
        if (link_list[cp[0]] > -1) and (link_list[cp[1]] > -1):
            if (link_list[cp[0]], link_list[cp[1]]) in line_gt:
                cnt_pred += 1

    if score_type == 'Graph':
        tp = cnt_obj + cnt_pred
        n_recall = len(tower_gt) + len(line_gt)
        n_precision = len(tower_pred) + len(line_pred)
    elif score_type == 'Tower':
        tp = cnt_obj
        n_recall = len(tower_gt)
        n_precision = len(tower_pred)
    else:
        tp = cnt_pred
        n_recall = len(line_gt)
        n_precision = len(line_pred)

    return tp, n_recall, n_precision


def find_point_id(centers, point):
    dist = np.linalg.norm(np.array(centers) - np.array(point), axis=1)
    #assert np.min(dist) < 100
    return np.argmin(dist)

def get_city_name(file_name):
    seps = file_name.split('_')
    for s in seps:
        if len(s) > 3:
            return s

def get_city_id(file_name):
    seps = file_name.split('_')
    for s in seps:
        int_s = ''.join([a for a in s if a.isdigit()])
        if len(int_s) > 0:
            return int(int_s)


def get_reannotate_results(data_dir):
    return natsorted(glob(os.path.join(data_dir, '*.csv')))


def get_annotate_results(re_files, data_dir):
    files = []
    for rf in re_files:
        file_name = os.path.basename(rf)
        city_name = get_city_name(file_name)
        city_id = get_city_id(file_name)
        if city_name == 'Colwich':
            city_name = 'Colwich_Maize'
        gt_file = glob(os.path.join(data_dir, '*_{}_{}.csv'.format(city_name, city_id))) + \
                  glob(os.path.join(data_dir, '*_{}_{}_resize.csv'.format(city_name, city_id)))
        assert len(gt_file) == 1
        files.append(gt_file[0])
    return files


def read_tower_truth(csv_file_name):
    gt_list = []
    for label, bbox in read_polygon_csv_data(csv_file_name):
        y, x = get_center_point(*bbox)
        gt_list.append([y, x])
    return gt_list


def read_lines_truth(csv_file_name, tower_gt):
    connected_pair = []
    for start, stop, online_points in read_line_csv_data(csv_file_name, tower_gt):
        centers = add_point_if_not_nearby(start, stop, [tower_gt[a] for a in online_points])
        for i in range(len(centers) - 1):
            # find the corresponding gt
            try:
                connected_pair.append(order_pair(find_point_id(tower_gt, centers[i]),
                                                 find_point_id(tower_gt, centers[i + 1])))
            except AssertionError:
                pass
    return connected_pair


def check_gt_plot(city_name, city_id, tower_gt):
    if city_name == 'Colwich':
        city_name = 'Colwich_Maize'
    elif city_name == 'Palmerston North':
        city_name = 'PalmerstonNorth'
    elif city_name == 'USA_AZ_Tucson':
        city_name = 'Tucson'
    else:
        city_name = city_id.split('_')[1]
    rgb_file = os.path.join(r'/media/ei-edl01/data/uab_datasets/lines_v3/data/Original_Tiles',
                            '{}{}_RGB.tif'.format(city_name, city_id))

    try:
        line_gt = misc_utils.load_file(os.path.join(r'/media/ei-edl01/data/uab_datasets/lines_v3/data/Original_Tiles',
                                                    '{}{}_GT.png'.format(city_name, city_id)))
        line_gt = cv2.dilate(line_gt, np.ones((5, 5), np.uint8), iterations=3)

        plt.figure(figsize=(8, 6))
        rgb = misc_utils.load_file(rgb_file)
        rgb = util_functions.add_mask(rgb, line_gt, [0, 255, 0], 1)
        plt.imshow(rgb)
        # visualize_with_connected_pairs(rgb, tower_gt, line_gt, style='r', add_fig=True)
        center_points = np.array(tower_gt)
        plt.scatter(center_points[:, 1], center_points[:, 0], c='g', s=40, marker='o', alpha=1,
                    edgecolors='k')
        plt.title('{} {}'.format(city_name, city_id))
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    except:
        print(rgb_file)


def check_gt_plot2(rgb_file, city_name, city_id, tower_gt):
    if city_name == 'USA_KS_Colwich_Maize':
        city_name = 'Colwich_Maize'
    if city_name == 'NZ_Palmerston North':
        city_name = 'PalmerstonNorth'
    try:
        line_gt = misc_utils.load_file(os.path.join(r'/media/ei-edl01/data/uab_datasets/lines_v3/data/Original_Tiles',
                                                    '{}{}_GT.png'.format(city_name, city_id)))
        line_gt = cv2.dilate(line_gt, np.ones((5, 5), np.uint8), iterations=3)

        plt.figure(figsize=(8, 6))
        rgb = misc_utils.load_file(rgb_file)
        rgb = util_functions.add_mask(rgb, line_gt, [0, 255, 0], 1)
        plt.imshow(rgb)
        # visualize_with_connected_pairs(rgb, tower_gt, line_gt, style='r', add_fig=True)
        center_points = np.array(tower_gt)
        plt.scatter(center_points[:, 1], center_points[:, 0], c='g', s=40, marker='o', alpha=1,
                    edgecolors='k')
        plt.title('{} {}'.format(city_name, city_id))
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    except:
        pass


def plot_error_bar(city_f1, city_p, city_r, title):
    city_list = sorted(list(city_f1.keys()))
    f1_list = []
    p_list = []
    r_list = []
    f1_yerr_min_list = []
    f1_yerr_max_list = []
    p_yerr_min_list = []
    p_yerr_max_list = []
    r_yerr_min_list = []
    r_yerr_max_list = []
    for city_name in city_list:
        f1_list.append(np.mean(city_f1[city_name]))
        f1_yerr_min_list.append(np.mean(city_f1[city_name]) - np.min(city_f1[city_name]))
        f1_yerr_max_list.append(np.max(city_f1[city_name]) - np.mean(city_f1[city_name]))

        p_list.append(np.mean(city_f1[city_name]))
        p_yerr_min_list.append(np.mean(city_p[city_name]) - np.min(city_p[city_name]))
        p_yerr_max_list.append(np.max(city_p[city_name]) - np.mean(city_p[city_name]))

        r_list.append(np.mean(city_r[city_name]))
        r_yerr_min_list.append(np.mean(city_r[city_name]) - np.min(city_r[city_name]))
        r_yerr_max_list.append(np.max(city_r[city_name]) - np.mean(city_r[city_name]))
    f1_yerr = np.stack((f1_yerr_min_list, f1_yerr_max_list), axis=0)
    p_yerr = np.stack((p_yerr_min_list, p_yerr_max_list), axis=0)
    r_yerr = np.stack((r_yerr_min_list, r_yerr_max_list), axis=0)

    city_num = len(city_list)
    plt.figure(figsize=(6, 4))
    n = np.arange(city_num)
    width = 0.3
    plt.bar(n, p_list, width=width, yerr=p_yerr, label='Precision')
    plt.bar(n + width, r_list, width=width, yerr=r_yerr, label='Recall')
    plt.bar(n + width * 2, f1_list, width=width, yerr=f1_yerr, label='F1')
    plt.xticks(n+width, city_list)
    plt.xlabel('City Name')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.ylim([0, 1])
    plt.title(title)
    plt.tight_layout()
    # plt.show()


if __name__ == '__main__':
    # cities
    city_name = ['NZ_Dunedin', 'NZ_Gisborne', 'NZ_Palmerston North', 'NZ_Rotorua',
                 'NZ_Tauranga', 'USA_AZ_Tucson', 'USA_KS_Colwich_Maize']
    resize_flag = [1, 1, 1, 1, 1, 0, 0]
    city_num = [6, 6, 17, 8, 8, 12, 15]

    # settings
    data_dir = r'/home/lab/Documents/bohao/data/transmission_line/raw2'
    img_dir, task_dir = sis_utils.get_task_img_folder()

    for c_name, r_flag, c_num in zip(city_name, resize_flag, city_num):
        for city_id in range(1, c_num+1):
            if r_flag == 1:
                rgb_name = '{}_{}_resize.tif'.format(c_name, city_id)
                gt_name = '{}_{}_resize.csv'.format(c_name, city_id)
            else:
                rgb_name = '{}_{}.tif'.format(c_name, city_id)
                gt_name = '{}_{}.csv'.format(c_name, city_id)

            try:
                t_gt = read_tower_truth(os.path.join(data_dir, gt_name))
                check_gt_plot2(os.path.join(data_dir, rgb_name), c_name, city_id, t_gt)
            except FileNotFoundError:
                print(gt_name)
