"""

"""


# Built-in
import os

# Libs
import scipy.spatial
import numpy as np

# Own modules
import sis_utils
import ersa_utils
from post_processing_utils import read_polygon_csv_data
from evaluate_utils import get_center_point, local_maxima_suppression


def radius_scoring_f1(pred, gt, link_r=20):
    # link predictions
    kdt = scipy.spatial.KDTree(gt)
    d, linked_results = kdt.query(pred)

    tp, fp, fn = 0, 0, 0
    linked_preds = []
    for cnt, item in enumerate(linked_results):
        if d[cnt] > link_r:
            # no gt, false pasitive
            fp += 1
        else:
            # there is at least one gt linked
            tp += 1
            linked_preds.append(item)  # store preds to calculate fp
    linked_preds = np.unique(linked_preds).tolist()
    for item in range(len(gt)):
        if item not in linked_preds:
            # no pred linked to gt, false negative
            fn += 1
    p = tp / (tp + fn)
    r = tp / (tp + fp)
    return 2 * p * r / (p + r), tp, tp + fn, tp + fp


def compare_towers(task_dir, raw_dir):
    city_list = ['AZ_Tucson', 'KS_Colwich_Maize', 'NC_Clyde', 'NC_Wilmington']
    model_names = ['faster_rcnn_res101', 'faster_rcnn_res50', 'faster_rcnn']

    for model_cnt, model_name in enumerate(model_names):
        print(model_name)
        tp_all, p_d_all, r_d_all = 0, 0, 0
        for city_id in range(4):
            pred_list_all = []
            gt_list_all = []
            for tile_id in [1, 2, 3]:
                # load data
                pred_file_name = os.path.join(task_dir, model_name+'_all',
                                              'USA_{}_{}.txt'.format(city_list[city_id], tile_id))
                preds = ersa_utils.load_file(pred_file_name)
                csv_file_name = os.path.join(raw_dir, 'USA_{}_{}.csv'.format(city_list[city_id], tile_id))

                pred_list = []
                gt_list = []
                center_list, conf_list, _ = local_maxima_suppression(preds)
                for center, conf in zip(center_list, conf_list):
                    pred_list.append(center.tolist())

                for label, bbox in read_polygon_csv_data(csv_file_name):
                    y, x = get_center_point(*bbox)
                    gt_list.append([y, x])

                pred_list_all.extend(pred_list)
                gt_list_all.extend(gt_list)
            f1, tp, p_d, r_d = radius_scoring_f1(pred_list_all, gt_list_all)
            tp_all += tp
            p_d_all += p_d
            r_d_all += r_d
            print('{}: f1={:.2f}'.format(city_list[city_id], f1))
        p = tp_all / p_d_all
        r = tp_all / r_d_all
        f1 = 2 * p * r / (p + r)
        print('overall f1={:.2f}'.format(f1))


def adjust_cp(cp):
    if cp[0] > cp[1]:
        return np.array([cp[1], cp[0]])
    else:
        return cp


def grid_score(cp_pred, cp_gt):
    cp_pred = set(tuple(cp) for cp in cp_pred)
    cp_pred = [list(cp) for cp in cp_pred]
    tp = 0
    for cp in cp_pred:
        for gt in cp_gt:
            if np.all([adjust_cp(cp) == gt]):
                tp += 1
    p_d = len(cp_pred)
    r_d = len(cp_gt)
    p = tp / (p_d + 1e-5)
    r = tp / (r_d + 1e-5)
    f1 = 2 * (p * r) / (p + r + 1e-5)
    return f1, tp, p_d, r_d


def compare_lines_base(task_dir):
    city_list = ['AZ_Tucson', 'KS_Colwich_Maize', 'NC_Clyde', 'NC_Wilmington']
    tp_all, p_d_all, r_d_all = 0, 0, 0
    for city_id in range(4):
        tp_city = 0
        pd_city = 0
        rd_city = 0
        for tile_id in [1, 2, 3]:
            cp_file_name = os.path.join(task_dir, '{}_{}_cp.npy'.format(city_list[city_id], city_id))
            connected_pairs = ersa_utils.load_file(cp_file_name)

            conns_name = 'overall_post_gt_{}_{}_conn2.npy'.format(city_id, tile_id)
            conns = np.load(os.path.join(task_dir, conns_name))

            _, tp, p_d, r_d = grid_score(conns, connected_pairs)
            tp_city += tp
            pd_city += p_d
            rd_city += r_d
        p = tp_city / pd_city
        r = tp_city / rd_city
        f1 = 2 * p * r / (p + r)
        print('{}: f1={:.2f}'.format(city_list[city_id], f1))

        tp_all += tp_city
        p_d_all += pd_city
        r_d_all += rd_city
    p = tp_all / p_d_all
    r = tp_all / r_d_all
    f1 = 2 * p * r / (p + r)
    print('overall f1={:.2f}'.format(f1))


def compare_lines_rnn(task_dir):
    city_list = ['AZ_Tucson', 'KS_Colwich_Maize', 'NC_Clyde', 'NC_Wilmington']
    tp_all, p_d_all, r_d_all = 0, 0, 0
    for city_id in range(4):
        tp_city = 0
        pd_city = 0
        rd_city = 0
        for tile_id in [1, 2, 3]:
            cp_file_name = os.path.join(task_dir, '{}_{}_cp.npy'.format(city_list[city_id], city_id))
            connected_pairs = ersa_utils.load_file(cp_file_name)

            conns_name = os.path.join(task_dir, 'graph_rnn_{}_{}_gt.npy'.format(city_id, tile_id))
            conns = np.load(os.path.join(task_dir, conns_name))

            _, tp, p_d, r_d = grid_score(conns, connected_pairs)
            tp_city += tp
            pd_city += p_d
            rd_city += r_d
        p = tp_city / pd_city
        r = tp_city / rd_city
        f1 = 2 * p * r / (p + r)
        print('{}: f1={:.2f}'.format(city_list[city_id], f1))

        tp_all += tp_city
        p_d_all += pd_city
        r_d_all += rd_city
    p = tp_all / p_d_all
    r = tp_all / r_d_all
    f1 = 2 * p * r / (p + r)
    print('overall f1={:.2f}'.format(f1))


if __name__ == '__main__':
    img_dir, task_dir = sis_utils.get_task_img_folder()
    raw_dir = r'/home/lab/Documents/bohao/data/transmission_line/raw'
    # compare_towers(task_dir, raw_dir)
    compare_lines_base(task_dir)
    compare_lines_rnn(task_dir)
