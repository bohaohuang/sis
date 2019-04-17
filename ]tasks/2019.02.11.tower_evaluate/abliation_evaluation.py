import os
import scipy.spatial
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.draw import polygon
import sis_utils
import ersa_utils
from evaluate_utils import get_center_point, local_maxima_suppression
from post_processing_utils import order_pair


def find_close_points(p1, p2, points, th=10):
    def dist2line(p1, p2, p3):
        p1, p2, p3 = np.array(p1), np.array(p2), np.array(p3)
        return np.linalg.norm(np.cross(p2 - p1, p3 - p1))/np.linalg.norm(p2 - p1)

    def is_between(p1, p2, p3):
        p1, p2, p3 = np.array(p1), np.array(p2), np.array(p3)
        if 1 - scipy.spatial.distance.cosine((p1 - p3), (p2 - p3)) < 0:
            return True
        else:
            return False

    point_ids = []
    for cnt, p3 in enumerate(points):
        if dist2line(p1, p2, p3[::-1]) < th and is_between(p1, p2, p3[::-1]):
            point_ids.append(cnt)

    return point_ids


def read_line_csv_data(csv_file, centers):
    label_order = ['SS', 'OT', 'DT', 'TT', 'OL', 'DL', 'TL']
    df = pd.read_csv(csv_file)
    df['temp_label'] = pd.Categorical(df['Label'], categories=label_order, ordered=True)
    df.sort_values('temp_label', inplace=True, kind='mergesort')

    for name, group in df.groupby('Object', sort=False):
        label = group['Label'].values[0]
        if group['Type'].values[0] == 'Line' and label in ['TL', 'DL']:
            for j in range(group.shape[0] - 1):
                r0, c0 = int(group['X'].values[j]), int(group['Y'].values[j])
                r1, c1 = int(group['X'].values[j + 1]), int(group['Y'].values[j + 1])
                p1 = (r0, c0)
                p2 = (r1, c1)
                online_points = find_close_points(p1, p2, centers)
                yield (r0, c0), (r1, c1), online_points


def add_point_if_not_nearby(p1, p2, points, th=20):
    p1 = np.array(p1)
    p2 = np.array(p2)
    p1_flag = True
    p2_flag = True
    for p3 in points:
        p3 = np.array(p3[::-1])
        if np.linalg.norm(p3 - p1) < th:
            p1_flag = False
        if np.linalg.norm(p3 - p2) < th:
            p2_flag = False
    if p1_flag:
        points.append(p1.tolist()[::-1])
    if p2_flag:
        points.append(p2.tolist()[::-1])
    return points


def find_point_id(centers, point):
    dist = np.linalg.norm(np.array(centers) - np.array(point), axis=1)
    #assert np.min(dist) < 100
    return np.argmin(dist)


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


def read_polygon_csv_data(csv_file):
    def get_bounding_box(y, x):
        y_min = np.min(y).astype(int)
        x_min = np.min(x).astype(int)
        y_max = np.max(y).astype(int)
        x_max = np.max(x).astype(int)
        return y_min, x_min, y_max, x_max

    encoder = {'DT': 1, 'TT': 2, 'T': 1}
    label_order = ['SS', 'OT', 'DT', 'TT', 'OL', 'DL', 'TL']
    df = pd.read_csv(csv_file)
    df['temp_label'] = pd.Categorical(df['Label'], categories=label_order, ordered=True)
    df.sort_values('temp_label', inplace=True, kind='mergesort')

    for name, group in df.groupby('Object', sort=False):
        label = group['Label'].values[0]
        if group['Type'].values[0] == 'Polygon' and label in encoder:
            x, y = polygon(group['X'].values, group['Y'].values)
            yield label, get_bounding_box(y, x)


def custom_scoring(pred, gt, confidences, link_r):
    # link predictions
    kdt = scipy.spatial.KDTree(pred)
    linked_results = kdt.query_ball_point(gt, link_r)
    y_true, y_score = [], []

    tp, fp, fn = 0, 0, 0
    linked_preds = []
    for item in linked_results:
        if not item:
            # no linked preds, false negative
            fn += 1
            y_true.append(1)
            y_score.append(0)
        else:
            # there is at least one pred linked
            tp += 1
            y_true.append(1)
            linked_preds.extend(item)  # store preds to calculate fp
            y_score.append(confidences[item[0]])
            for false_positive in item[1:]:
                fp += 1  # redundant bboxes
                y_true.append(0)
                y_score.append(confidences[false_positive])
    linked_preds = np.unique(linked_preds).tolist()
    for item in range(len(pred)):
        if item not in linked_preds:
            # preds that are not linked
            fp += 1
            y_true.append(0)
            y_score.append(confidences[item])
    p = tp / (tp + fn)
    r = tp / (tp + fp)
    return 2 * p * r / (p + r), y_true, y_score


def radius_scoring(pred, gt, confidences, link_r):
    # link predictions
    kdt = scipy.spatial.KDTree(gt)
    d, linked_results = kdt.query(pred)
    y_true, y_score = [], []

    tp, fp, fn = 0, 0, 0
    linked_preds = []
    for cnt, item in enumerate(linked_results):
        if d[cnt] > link_r:
            # no gt, false pasitive
            fp += 1
            y_true.append(0)
            y_score.append(confidences[cnt])
        else:
            # there is at least one gt linked
            tp += 1
            y_true.append(1)
            linked_preds.append(item)  # store preds to calculate fp
            y_score.append(confidences[cnt])
    linked_preds = np.unique(linked_preds).tolist()
    for item in range(len(gt)):
        if item not in linked_preds:
            # no pred linked to gt, false negative
            fn += 1
            y_true.append(1)
            y_score.append(0)
    p = tp / (tp + fn)
    r = tp / (tp + fp)
    return 2 * p * r / (p + r), y_true, y_score


def link_pred_gt(pred, gt, link_r):
    # link predictions
    kdt = scipy.spatial.KDTree(gt)
    d, linked_results = kdt.query(pred)
    link_list = [-1 for _ in range(len(pred))]
    be_linked = {}

    for cnt, item in enumerate(linked_results):
        if d[cnt] > link_r:
            pass
        else:
            be_linked_item = int(item)
            if be_linked_item not in be_linked:
                link_list[cnt] = be_linked_item
                be_linked[be_linked_item] = [d[cnt], cnt]
            else:
                if d[cnt] < be_linked[be_linked_item][0]:
                    link_list[be_linked[be_linked_item][1]] = -1
                    link_list[cnt] = be_linked_item
                    be_linked[be_linked_item] = [d[cnt], cnt]
    return link_list


def grid_score(tower_gt, tower_pred, line_gt, line_pred, link_list):
    cnt_obj = 0
    for a in link_list:
        if a > -1:
            cnt_obj += 1
    cnt_pred = 0

    lp = []
    for cp in line_pred:
        lp.append(order_pair(*cp))
    lp = list(set(lp))

    for cp in lp:
        if (link_list[cp[0]] > -1) and (link_list[cp[1]] > -1):
            if (link_list[cp[0]], link_list[cp[1]]) in line_gt:
                cnt_pred += 1

    tp = cnt_obj + cnt_pred
    n_recall = len(tower_gt) + len(line_gt)
    n_precision = len(tower_pred) + len(line_pred)

    return tp, n_recall, n_precision


def plot_across_model_grid_with_graph(link_r=20, model_names=('faster_rcnn', 'faster_rcnn_res101', 'faster_rcnn_res50')):
    plt.figure(figsize=(14, 8))
    model_name_dict = {
        'faster_rcnn': 'Inception',
        'faster_rcnn_res101': 'ResNet101',
        'faster_rcnn_res50': 'ResNet50'
    }

    city_list = ['AZ_Tucson', 'KS_Colwich_Maize', 'NC_Clyde', 'NC_Wilmington']
    # cmap = ersa_utils.get_default_colors()
    cmap = plt.get_cmap('tab20')
    width = 0.1

    model_n_tp = np.zeros((3, len(model_names)))
    model_n_recall = np.zeros((3, len(model_names)))
    model_n_precision = np.zeros((3, len(model_names)))

    for city_id in range(4):
        plt.subplot(221 + city_id)
        for model_cnt, model_name in enumerate(model_names):
            tp_all, tp_all_graph, tp_all_graph_combine = 0, 0, 0
            n_recall_all, n_recall_all_graph, n_recall_all_graph_combine = 0, 0, 0
            n_precision_all, n_precision_all_graph, n_precision_all_graph_combine = 0, 0, 0

            for tile_id in [1, 2, 3]:
                # load data
                pred_file_name = os.path.join(task_dir, model_name, 'USA_{}_{}.txt'.format(city_list[city_id], tile_id))
                preds = ersa_utils.load_file(pred_file_name)
                pred_file_name = os.path.join(task_dir, 'w1_post_{}_{}_{}_pred2.npy'.format(model_name, city_id, tile_id))
                pred_list = ersa_utils.load_file(pred_file_name)
                pred_file_name = os.path.join(task_dir, 'w1_post_{}_{}_{}_conn2.npy'.format(model_name, city_id, tile_id))
                cp_list = ersa_utils.load_file(pred_file_name)
                csv_file_name = os.path.join(raw_dir, 'USA_{}_{}.csv'.format(city_list[city_id], tile_id))
                #cp_file_name = os.path.join(task_dir, '{}_{}_cp.npy'.format(city_list[city_id], city_id))
                #connected_pairs = ersa_utils.load_file(cp_file_name)

                cp_file_name_graph = os.path.join(task_dir, '{}_graph_rnn_{}_{}_orig.npy'.format(model_name, city_id, tile_id))
                cp_graph = ersa_utils.load_file(cp_file_name_graph)

                cp_file_name_graph = os.path.join(task_dir, '{}_graph_rnn_{}_{}.npy'.format(model_name, city_id, tile_id))
                cp_graph_combine = ersa_utils.load_file(cp_file_name_graph)

                tower_gt = []
                for label, bbox in read_polygon_csv_data(csv_file_name):
                    y, x = get_center_point(*bbox)
                    tower_gt.append([y, x])
                line_file = os.path.join(raw_dir, 'USA_{}_{}.csv'.format(city_list[city_id], tile_id))
                connected_pairs = read_lines_truth(line_file, tower_gt)

                pred_list_orig = []
                center_list, conf_list, _ = local_maxima_suppression(preds)
                for center, conf in zip(center_list, conf_list):
                    if conf > 0.5:
                        pred_list_orig.append(center.tolist())

                link_list = link_pred_gt(pred_list, tower_gt, link_r)
                tp, n_recall, n_precision = grid_score(tower_gt, pred_list, connected_pairs, cp_list, link_list)

                tp_graph_combine, n_recall_graph_combine, n_precision_graph_combine = \
                    grid_score(tower_gt, pred_list, connected_pairs, cp_graph_combine, link_list)

                link_list = link_pred_gt(center_list, tower_gt, link_r)
                tp_graph, n_recall_graph, n_precision_graph = grid_score(tower_gt, pred_list, connected_pairs, cp_graph,
                                                                         link_list)

                tp_all += tp
                n_recall_all += n_recall
                n_precision_all += n_precision
                tp_all_graph += tp_graph
                n_recall_all_graph += n_recall_graph
                n_precision_all_graph += n_precision_graph
                tp_all_graph_combine += tp_graph_combine
                n_recall_all_graph_combine += n_recall_graph_combine
                n_precision_all_graph_combine += n_precision_graph_combine

                model_n_tp[0, model_cnt] += tp
                model_n_tp[1, model_cnt] += tp_graph
                model_n_tp[2, model_cnt] += tp_graph_combine
                model_n_recall[0, model_cnt] += n_recall
                model_n_recall[1, model_cnt] += n_recall_graph
                model_n_recall[2, model_cnt] += n_recall_graph_combine
                model_n_precision[0, model_cnt] += n_precision
                model_n_precision[1, model_cnt] += n_precision_graph
                model_n_precision[2, model_cnt] += n_precision_graph_combine

            recall = tp_all / n_recall_all
            precision = tp_all / n_precision_all
            f1 = 2 * (precision * recall) / (precision + recall)
            recall_graph = tp_all_graph / n_recall_all_graph
            precision_graph = tp_all_graph / n_precision_all_graph
            f1_graph = 2 * (precision_graph * recall_graph) / (precision_graph + recall_graph)
            recall_graph_combine = tp_all_graph_combine / n_recall_all_graph_combine
            precision_graph_combine = tp_all_graph_combine / n_precision_all_graph_combine
            f1_graph_combine = 2 * (precision_graph_combine * recall_graph_combine) / (precision_graph_combine + recall_graph_combine)

    recall_overall = model_n_tp / model_n_recall
    precision_overall = model_n_tp / model_n_precision
    f1_overall = 2 * (recall_overall * precision_overall) / (recall_overall + precision_overall)
    print(f1_overall)


def ablation_tower(model_name, link_r=20):
    city_list = ['AZ_Tucson', 'KS_Colwich_Maize', 'NC_Clyde', 'NC_Wilmington']

    model_n_tp = 0
    model_n_recall = 0
    model_n_precision = 0

    for city_id in range(4):
        tp_all, n_precision_all, n_recall_all = 0, 0, 0

        for tile_id in [1, 2, 3]:
            # load data
            pred_file_name = os.path.join(task_dir, model_name, 'USA_{}_{}.txt'.format(city_list[city_id], tile_id))
            preds = ersa_utils.load_file(pred_file_name)
            pred_file_name = os.path.join(task_dir, 'post_{}_{}_{}_pred3.npy'.format(model_name, city_id, tile_id))
            pred_list = ersa_utils.load_file(pred_file_name)
            pred_file_name = os.path.join(task_dir, 'post_{}_{}_{}_conn3.npy'.format(model_name, city_id, tile_id))
            cp_list = ersa_utils.load_file(pred_file_name)
            csv_file_name = os.path.join(raw_dir, 'USA_{}_{}.csv'.format(city_list[city_id], tile_id))

            tower_gt = []
            for label, bbox in read_polygon_csv_data(csv_file_name):
                y, x = get_center_point(*bbox)
                tower_gt.append([y, x])
            line_file = os.path.join(raw_dir, 'USA_{}_{}.csv'.format(city_list[city_id], tile_id))
            connected_pairs = read_lines_truth(line_file, tower_gt)

            pred_list_orig = []
            center_list, conf_list, _ = local_maxima_suppression(preds)
            for center, conf in zip(center_list, conf_list):
                if conf > 0.5:
                    pred_list_orig.append(center.tolist())

            if len(pred_list != 0):
                link_list = link_pred_gt(pred_list, tower_gt, link_r)
                tp, n_recall, n_precision = grid_score(tower_gt, center_list, connected_pairs, cp_list, link_list)
            else:
                tp = 0
                n_recall = 0
                n_precision = 0

            tp_all += tp
            n_recall_all += n_recall
            n_precision_all += n_precision

            model_n_tp += tp
            model_n_recall += n_recall
            model_n_precision += n_precision

        p = tp_all / n_precision_all
        r = tp_all / n_recall_all
        f1 = 2 * (p * r) / (p + r)
        print('{}: f1={:.2f}'.format(city_list[city_id], f1))

    recall_overall = model_n_tp / model_n_recall
    precision_overall = model_n_tp / model_n_precision
    f1_overall = 2 * (recall_overall * precision_overall) / (recall_overall + precision_overall)
    print('Overall: f1={:.2f}'.format(f1_overall))


def ablation_line(model_name, link_r=20):
    city_list = ['AZ_Tucson', 'KS_Colwich_Maize', 'NC_Clyde', 'NC_Wilmington']

    model_n_tp = 0
    model_n_recall = 0
    model_n_precision = 0

    for city_id in range(4):
        tp_all, n_precision_all, n_recall_all = 0, 0, 0

        for tile_id in [1, 2, 3]:
            # load data
            pred_file_name = os.path.join(task_dir, model_name, 'USA_{}_{}.txt'.format(city_list[city_id], tile_id))
            preds = ersa_utils.load_file(pred_file_name)
            csv_file_name = os.path.join(raw_dir, 'USA_{}_{}.csv'.format(city_list[city_id], tile_id))

            cp_file_name_graph = os.path.join(task_dir, '{}_graph_rnn_normal_{}_{}.npy'.format(model_name, city_id, tile_id))
            cp_graph = ersa_utils.load_file(cp_file_name_graph)

            tower_gt = []
            for label, bbox in read_polygon_csv_data(csv_file_name):
                y, x = get_center_point(*bbox)
                tower_gt.append([y, x])
            line_file = os.path.join(raw_dir, 'USA_{}_{}.csv'.format(city_list[city_id], tile_id))
            connected_pairs = read_lines_truth(line_file, tower_gt)

            pred_list_orig = []
            center_list, conf_list, _ = local_maxima_suppression(preds)
            for center, conf in zip(center_list, conf_list):
                if conf > 0.5:
                    pred_list_orig.append(center.tolist())

            if len(center_list) != 0:
                link_list = link_pred_gt(center_list, tower_gt, link_r)
                tp, n_recall, n_precision = grid_score(tower_gt, center_list, connected_pairs, cp_graph, link_list)
            else:
                tp = 0
                n_recall = 0
                n_precision = 0

            tp_all += tp
            n_recall_all += n_recall
            n_precision_all += n_precision

            model_n_tp += tp
            model_n_recall += n_recall
            model_n_precision += n_precision

        p = tp_all / n_precision_all
        r = tp_all / n_recall_all
        f1 = 2 * (p * r) / (p + r)
        print('{}: f1={:.2f}'.format(city_list[city_id], f1))

    recall_overall = model_n_tp / model_n_recall
    precision_overall = model_n_tp / model_n_precision
    f1_overall = 2 * (recall_overall * precision_overall) / (recall_overall + precision_overall)
    print('Overall: f1={:.2f}'.format(f1_overall))


# settings
if __name__ == '__main__':
    img_dir, task_dir = sis_utils.get_task_img_folder()

    data_dir = r'/home/lab/Documents/bohao/data/transmission_line'
    info_dir = os.path.join(data_dir, 'info')
    raw_dir = os.path.join(data_dir, 'raw')

    #plot_across_model_post()
    #plot_within_model('faster_rcnn')
    ablation_tower(model_name='faster_rcnn')
    # ablation_line(model_name='faster_rcnn_res50')
