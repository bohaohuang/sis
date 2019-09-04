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
import util_functions
from rst_utils import misc_utils
from evaluate_tower_performance import link_pred_gt
from line_length_stats import read_line_csv_data, add_point_if_not_nearby
from post_processing_utils import read_polygon_csv_data, get_center_point, order_pair, visualize_with_connected_pairs


def grid_score(tower_gt, tower_pred, line_gt, line_pred, link_list, type='all'):
    assert type in ['all', 'tower', 'line']

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

    if type == 'all':
        tp = cnt_obj + cnt_pred
        n_recall = len(tower_gt) + len(line_gt)
        n_precision = len(tower_pred) + len(line_pred)
    elif type == 'tower':
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


def get_reannotate_results(data_dir):
    return natsorted(glob(os.path.join(data_dir, '*.csv')))


def get_annotate_results(re_files, data_dir):
    files = []
    for rf in re_files:
        file_name = os.path.basename(rf)
        city_name = file_name.split('_')[0]
        city_id = str(file_name.split('_')[1]).split('.')[0]
        if city_name == 'Colwich':
            city_name = 'Colwich_Maize'
        gt_file = glob(os.path.join(data_dir, '*_{}_{}.csv'.format(city_name, city_id)))
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
            except ValueError:
                pass
    return connected_pair


def check_gt_plot(city_name, city_id, tower_gt, line_gt):
    if city_name == 'Colwich':
        city_name = 'Colwich_Maize'
    img_dir = r'/home/lab/Documents/bohao/data/transmission_line/raw'
    rgb_file = glob(os.path.join(img_dir, '*_{}_{}.tif'.format(city_name, city_id)))

    line_gt = misc_utils.load_file(os.path.join(r'/media/ei-edl01/data/uab_datasets/lines/data/Original_Tiles',
                                                '{}{}_GT.png'.format(city_name, city_id)))
    line_gt = cv2.dilate(line_gt, np.ones((5, 5), np.uint8), iterations=10)

    plt.figure(figsize=(8, 6))
    assert len(rgb_file) == 1
    rgb = misc_utils.load_file(rgb_file[0])
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


def plot_error_bar(city_f1, city_p, city_r, title):
    city_list = ['Tucson', 'Colwich', 'Clyde', 'Wilmington']
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
    city_num = len(city_list)
    f1_yerr = np.stack((f1_yerr_min_list, f1_yerr_max_list), axis=0)
    p_yerr = np.stack((p_yerr_min_list, p_yerr_max_list), axis=0)
    r_yerr = np.stack((r_yerr_min_list, r_yerr_max_list), axis=0)

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
    plt.show()


if __name__ == '__main__':
    # settings
    data_dir = r'/home/lab/Documents/bohao/data/transmission_line/annotate'
    gt_dir = r'/home/lab/Documents/bohao/data/transmission_line/raw'
    re_files = get_reannotate_results(data_dir)
    gt_files = get_annotate_results(re_files, gt_dir)

    city_f1 = {}
    city_p = {}
    city_r = {}

    for gt, re in zip(gt_files, re_files):
        print(gt, re)

        '''file_name = os.path.basename(re)
        city_name = file_name.split('_')[0]
        city_id = str(file_name.split('_')[1]).split('.')[0]

        t_gt = read_tower_truth(gt)
        t_re = read_tower_truth(re)
        cp_gt = read_lines_truth(gt, t_gt)
        cp_re = read_lines_truth(re, t_re)

        check_gt_plot(city_name, city_id, t_re, cp_re)

        link_list = link_pred_gt(t_re, t_gt, 20)

        tp, r_d, p_d = grid_score(t_gt, t_re, cp_gt, cp_re, link_list, type='line')
        p = tp / p_d
        r = tp / r_d
        f1 = 2 * (p * r) / (p + r)

        if city_name not in city_f1:
            city_f1[city_name] = [f1]
            city_p[city_name] = [p]
            city_r[city_name] = [r]
        else:
            city_f1[city_name].append(f1)
            city_p[city_name].append(p)
            city_r[city_name].append(r)

        # print('{}_{}: P:{:.2f}, R:{:.2f}, F1:{:.2f}'.format(city_name, city_id, p, r, f1))

    # plot error bar
    plot_error_bar(city_f1, city_p, city_r, 'Graph Level Comparison')'''
