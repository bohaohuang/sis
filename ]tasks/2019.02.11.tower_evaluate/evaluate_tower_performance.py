import os
import scipy.spatial
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage.draw import polygon
from sklearn.utils.fixes import signature
from sklearn.metrics import precision_recall_curve, average_precision_score
import utils
import ersa_utils
from evaluate_utils import get_center_point, local_maxima_suppression


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


# settings
if __name__ == '__main__':
    img_dir, task_dir = utils.get_task_img_folder()

    data_dir = r'/home/lab/Documents/bohao/data/transmission_line'
    info_dir = os.path.join(data_dir, 'info')
    raw_dir = os.path.join(data_dir, 'raw')

    plot_across_model()
    #plot_within_model('faster_rcnn')
