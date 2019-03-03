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


def get_linked_dist(pred, gt, link_r):
    # link predictions
    kdt = scipy.spatial.KDTree(pred)
    linked_results = kdt.query_ball_point(gt, link_r)
    dist = []

    for cnt, item in enumerate(linked_results):
        if not item:
            pass
        else:
            # TP
            dist.append(float(np.sqrt(np.sum(np.square(np.array(gt[cnt]) - np.array(pred[item[0]]))))))

    return dist


# settings
if __name__ == '__main__':
    img_dir, task_dir = sis_utils.get_task_img_folder()
    link_r = 60
    city_list = ['AZ_Tucson', 'KS_Colwich_Maize', 'NC_Clyde', 'NC_Wilmington']
    data_dir = r'/home/lab/Documents/bohao/data/transmission_line'
    info_dir = os.path.join(data_dir, 'info')
    raw_dir = os.path.join(data_dir, 'raw')
    dist = []

    for city_id in range(4):
        pred_list_all = []
        gt_list_all = []
        for tile_id in [1, 2, 3]:
            # load data
            pred_file_name = os.path.join(task_dir, 'USA_{}_{}.txt'.format(city_list[city_id], tile_id))
            preds = ersa_utils.load_file(pred_file_name)
            raw_rgb = ersa_utils.load_file(os.path.join(raw_dir, 'USA_{}_{}.tif'.format(city_list[city_id], tile_id)))
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

        dist.extend(get_linked_dist(pred_list_all, gt_list_all, link_r))

    plt.figure(figsize=(8, 4))
    plt.hist(dist, bins=200)
    dist_range = np.arange(0, 70, 5)
    plt.xticks(dist_range, dist_range*0.15)
    plt.xlabel('Dist(m)')
    plt.ylabel('Cnt')
    plt.title('Histogram of Linked Towers\' Distance')
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, 'dist_demo.png'))
    plt.show()
