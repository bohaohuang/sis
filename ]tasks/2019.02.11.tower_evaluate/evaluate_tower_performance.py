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


def parse_result(output_line):
    info = output_line.strip().split(' ')
    class_name = str(info[0])
    confidence = float(info[1])
    left = int(info[2])
    top = int(info[3])
    right = int(info[4])
    bottom = int(info[5])
    return class_name, confidence, left, top, right, bottom


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


def get_center_point(ymin, xmin, ymax, xmax):
    return ((ymin+ymax)/2, (xmin+xmax)/2)


def custom_scoring(pred, gt, confidences):
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


# settings
img_dir, task_dir = utils.get_task_img_folder()
city_id = 0
city_list = ['AZ_Tucson', 'KS_Colwich_Maize', 'NC_Clyde', 'NC_Wilmington']
data_dir = r'/home/lab/Documents/bohao/data/transmission_line'
info_dir = os.path.join(data_dir, 'info')
raw_dir = os.path.join(data_dir, 'raw')

plt.figure(figsize=(8, 6))
for link_r in [3, 33, 50, 167]:  # 10m range
    pred_list_all = []
    gt_list_all = []
    cf_list_all = []
    for tile_id in [1, 2, 3]:
        # load data
        pred_file_name = os.path.join(task_dir, 'USA_{}_{}.txt'.format(city_list[city_id], tile_id))
        preds = ersa_utils.load_file(pred_file_name)
        raw_rgb = ersa_utils.load_file(os.path.join(raw_dir, 'USA_{}_{}.tif'.format(city_list[city_id], tile_id)))
        csv_file_name = os.path.join(raw_dir, 'USA_{}_{}.csv'.format(city_list[city_id], tile_id))
        pred_list = []
        gt_list = []
        cf_list = []

        for line in preds:
            class_name, confidence, left, top, right, bottom = parse_result(line)
            y, x = get_center_point(top, left, bottom, right)
            pred_list.append([y, x])
            cf_list.append(confidence)
        for label, bbox in read_polygon_csv_data(csv_file_name):
            y, x = get_center_point(*bbox)
            gt_list.append([y, x])

        pred_list_all.extend(pred_list)
        gt_list_all.extend(gt_list)
        cf_list_all.extend(cf_list)

    f1, y_true, y_score = custom_scoring(pred_list_all, gt_list_all, cf_list_all)
    ap = average_precision_score(y_true, y_score)
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    step_kwargs = ({'step': 'post'}
                   if 'step' in signature(plt.fill_between).parameters
                   else {})
    plt.step(recall, precision, alpha=1, where='post', label='Link Radius={}, AP={:.2f}'.format(link_r, ap))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('{} Performance Comparison'.format(city_list[city_id]))

plt.legend()
plt.tight_layout()
plt.show()
