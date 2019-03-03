import os
import scipy.spatial
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage.draw import polygon
from sklearn.utils.fixes import signature
from sklearn.metrics import precision_recall_curve, average_precision_score
import sis_utils
import ersa_utils
from evaluate_utils import get_center_point, local_maxima_suppression
from post_processing_utils import order_pair


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

    for cnt, item in enumerate(linked_results):
        if d[cnt] > link_r:
            pass
        else:
            link_list[cnt] = int(item)
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


def plot_within_model(model_name='faster_rcnn'):
    plt.figure(figsize=(10, 8))

    city_list = ['AZ_Tucson', 'KS_Colwich_Maize', 'NC_Clyde', 'NC_Wilmington']
    for city_id in range(4):
        plt.subplot(221 + city_id)
        for link_r in [10, 20, 30, 60, 120, 240]:
            pred_list_all = []
            gt_list_all = []
            cf_list_all = []
            for tile_id in [1, 2, 3]:
                # load data
                pred_file_name = os.path.join(task_dir, model_name, 'USA_{}_{}.txt'.format(city_list[city_id], tile_id))
                preds = ersa_utils.load_file(pred_file_name)
                csv_file_name = os.path.join(raw_dir, 'USA_{}_{}.csv'.format(city_list[city_id], tile_id))
                pred_list = []
                gt_list = []
                cf_list = []

                center_list, conf_list, _ = local_maxima_suppression(preds)
                for center, conf in zip(center_list, conf_list):
                    pred_list.append(center.tolist())
                    cf_list.append(conf)

                for label, bbox in read_polygon_csv_data(csv_file_name):
                    y, x = get_center_point(*bbox)
                    gt_list.append([y, x])

                pred_list_all.extend(pred_list)
                gt_list_all.extend(gt_list)
                cf_list_all.extend(cf_list)

            f1, y_true, y_score = radius_scoring(pred_list_all, gt_list_all, cf_list_all, link_r)
            ap = average_precision_score(y_true, y_score)
            precision, recall, _ = precision_recall_curve(y_true, y_score)
            plt.step(recall[1:], precision[1:], alpha=1, where='post',
                     label='Link Radius={:.1f}m, AP={:.2f}'.format(link_r * 0.15, ap))
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.ylim([0.0, 1.05])
            plt.xlim([0.0, 1.0])
            plt.title('{} Performance Comparison'.format(city_list[city_id]))
            plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, '{}_tile_pr.png'.format(model_name)))
    plt.show()


def plot_across_model(link_r=20, model_names=('faster_rcnn', 'faster_rcnn_res101', 'faster_rcnn_res50')):
    plt.figure(figsize=(10, 8))

    city_list = ['AZ_Tucson', 'KS_Colwich_Maize', 'NC_Clyde', 'NC_Wilmington']
    for city_id in range(4):
        plt.subplot(221 + city_id)
        for model_name in model_names:
            pred_list_all = []
            gt_list_all = []
            cf_list_all = []
            for tile_id in [1, 2, 3]:
                # load data
                pred_file_name = os.path.join(task_dir, model_name, 'USA_{}_{}.txt'.format(city_list[city_id], tile_id))
                preds = ersa_utils.load_file(pred_file_name)
                csv_file_name = os.path.join(raw_dir, 'USA_{}_{}.csv'.format(city_list[city_id], tile_id))
                pred_list = []
                gt_list = []
                cf_list = []

                center_list, conf_list, _ = local_maxima_suppression(preds)
                for center, conf in zip(center_list, conf_list):
                    pred_list.append(center.tolist())
                    cf_list.append(conf)

                for label, bbox in read_polygon_csv_data(csv_file_name):
                    y, x = get_center_point(*bbox)
                    gt_list.append([y, x])

                pred_list_all.extend(pred_list)
                gt_list_all.extend(gt_list)
                cf_list_all.extend(cf_list)

            f1, y_true, y_score =radius_scoring(pred_list_all, gt_list_all, cf_list_all, link_r)
            ap = average_precision_score(y_true, y_score)
            precision, recall, _ = precision_recall_curve(y_true, y_score)
            plt.step(recall[1:], precision[1:], alpha=1, where='post',
                     label='{}, AP={:.2f}'.format(model_name, ap))
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.ylim([0.0, 1.05])
            plt.xlim([0.0, 1.0])
            plt.title('{} Performance Comparison'.format(city_list[city_id]))
            plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, 'cmp_tile_pr.png'))
    plt.show()


def plot_across_model_post(link_r=20, model_names=('faster_rcnn', 'faster_rcnn_res101', 'faster_rcnn_res50')):
    plt.figure(figsize=(10, 8))

    city_list = ['AZ_Tucson', 'KS_Colwich_Maize', 'NC_Clyde', 'NC_Wilmington']
    cmap = ersa_utils.get_default_colors()
    for city_id in range(4):
        plt.subplot(221 + city_id)
        for model_cnt, model_name in enumerate(model_names):
            pred_list_all_orig = []
            cf_list_all_orig = []
            pred_list_all = []
            cf_list_all = []
            gt_list_all = []
            for tile_id in [1, 2, 3]:
                # load data
                pred_file_name = os.path.join(task_dir, model_name, 'USA_{}_{}.txt'.format(city_list[city_id], tile_id))
                preds = ersa_utils.load_file(pred_file_name)

                pred_file_name = os.path.join(task_dir, 'post_{}_{}_{}_pred2.npy'.format(model_name, city_id, tile_id))
                pred_list = ersa_utils.load_file(pred_file_name)
                pred_file_name = os.path.join(task_dir, 'post_{}_{}_{}_conf2.npy'.format(model_name, city_id, tile_id))
                cf_list = ersa_utils.load_file(pred_file_name)
                csv_file_name = os.path.join(raw_dir, 'USA_{}_{}.csv'.format(city_list[city_id], tile_id))
                gt_list = []

                pred_list_orig = []
                cf_list_orig = []
                center_list, conf_list, _ = local_maxima_suppression(preds)
                for center, conf in zip(center_list, conf_list):
                    pred_list_orig.append(center.tolist())
                    cf_list_orig.append(conf)

                for label, bbox in read_polygon_csv_data(csv_file_name):
                    y, x = get_center_point(*bbox)
                    gt_list.append([y, x])

                pred_list_all_orig.extend(pred_list_orig)
                cf_list_all_orig.extend(cf_list_orig)
                pred_list_all.extend(pred_list)
                cf_list_all.extend(cf_list)
                gt_list_all.extend(gt_list)

            f1, y_true, y_score =radius_scoring(pred_list_all_orig, gt_list_all, cf_list_all_orig, link_r)
            ap = average_precision_score(y_true, y_score)
            precision, recall, _ = precision_recall_curve(y_true, y_score)
            plt.step(recall[1:], precision[1:], alpha=1, where='post', linestyle='--',
                     label='Orig {}, AP={:.2f}'.format(model_name, ap), color=cmap[model_cnt])

            f1, y_true, y_score = radius_scoring(pred_list_all, gt_list_all, cf_list_all, link_r)
            ap = average_precision_score(y_true, y_score)
            precision, recall, _ = precision_recall_curve(y_true, y_score)
            plt.step(recall[1:], precision[1:], alpha=1, where='post',
                     label='Post {}, AP={:.2f}'.format(model_name, ap), color=cmap[model_cnt])

            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.ylim([0.0, 1.05])
            plt.xlim([0.0, 1.0])
            plt.title('{} Performance Comparison'.format(city_list[city_id]))
            plt.legend(loc='lower left')
    plt.tight_layout()
    # plt.savefig(os.path.join(img_dir, 'cmp_tile_pr_orig_vs_post.png'))
    plt.show()


def plot_across_model_grid(link_r=20, model_names=('faster_rcnn', 'faster_rcnn_res101', 'faster_rcnn_res50')):
    plt.figure(figsize=(10, 8))
    model_name_dict = {
        'faster_rcnn': 'Inception',
        'faster_rcnn_res101': 'ResNet101',
        'faster_rcnn_res50': 'ResNet50'
    }

    city_list = ['AZ_Tucson', 'KS_Colwich_Maize', 'NC_Clyde', 'NC_Wilmington']
    cmap = ersa_utils.get_default_colors()
    width = 0.15
    for city_id in range(4):
        plt.subplot(221 + city_id)
        for model_cnt, model_name in enumerate(model_names):
            tp_all = 0
            n_recall_all = 0
            n_precision_all = 0

            for tile_id in [1, 2, 3]:
                # load data
                pred_file_name = os.path.join(task_dir, model_name, 'USA_{}_{}.txt'.format(city_list[city_id], tile_id))
                preds = ersa_utils.load_file(pred_file_name)

                pred_file_name = os.path.join(task_dir, 'post_{}_{}_{}_pred2.npy'.format(model_name, city_id, tile_id))
                pred_list = ersa_utils.load_file(pred_file_name)
                pred_file_name = os.path.join(task_dir, 'post_{}_{}_{}_conf2.npy'.format(model_name, city_id, tile_id))
                cf_list = ersa_utils.load_file(pred_file_name)
                pred_file_name = os.path.join(task_dir, 'post_{}_{}_{}_conn2.npy'.format(model_name, city_id, tile_id))
                cp_list = ersa_utils.load_file(pred_file_name)
                csv_file_name = os.path.join(raw_dir, 'USA_{}_{}.csv'.format(city_list[city_id], tile_id))
                cp_file_name = os.path.join(task_dir, '{}_{}_cp.npy'.format(city_list[city_id], city_id))
                connected_pairs = ersa_utils.load_file(cp_file_name)

                tower_gt = []
                for label, bbox in read_polygon_csv_data(csv_file_name):
                    y, x = get_center_point(*bbox)
                    tower_gt.append([y, x])

                pred_list_orig = []
                center_list, conf_list, _ = local_maxima_suppression(preds)
                for center, conf in zip(center_list, conf_list):
                    if conf > 0.5:
                        pred_list_orig.append(center.tolist())

                link_list = link_pred_gt(pred_list, tower_gt, link_r)
                tp, n_recall, n_precision = grid_score(tower_gt, pred_list, connected_pairs, cp_list, link_list)

                tp_all += tp
                n_recall_all += n_recall
                n_precision_all += n_precision

            recall = tp_all / n_recall_all
            precision = tp_all / n_precision_all

            X = np.arange(2)
            plt.bar(X+width*model_cnt, [precision, recall], width=width, color=cmap[model_cnt],
                    label=model_name_dict[model_name])
            plt.text(width*model_cnt-0.05, precision, '{:.2f}'.format(precision))
            plt.text(1+width*model_cnt-0.05, recall, '{:.2f}'.format(recall))

            #plt.xlabel('Recall')
            plt.xticks(X+width, ['Precision', 'Recall'])
            plt.ylabel('Score')
            plt.ylim([0.0, 1])
            #plt.xlim([0.0, 1.0])
            plt.title('{} Performance Comparison'.format(city_list[city_id]))
            plt.legend(loc='upper left', ncol=3, fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, 'tile_performance_graph.png'))
    plt.show()


# settings
if __name__ == '__main__':
    img_dir, task_dir = sis_utils.get_task_img_folder()

    data_dir = r'/home/lab/Documents/bohao/data/transmission_line'
    info_dir = os.path.join(data_dir, 'info')
    raw_dir = os.path.join(data_dir, 'raw')

    #plot_across_model_post()
    #plot_within_model('faster_rcnn')
    plot_across_model_grid()
