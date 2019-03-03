import os
import scipy.spatial
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sis_utils
import ersa_utils
from integrate_results_analysis import add_points
from evaluate_tower_performance import get_center_point, read_polygon_csv_data


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


if __name__ == '__main__':
    # directories
    img_dir, task_dir = sis_utils.get_task_img_folder()
    data_dir = r'/home/lab/Documents/bohao/data/transmission_line'
    raw_dir = os.path.join(data_dir, 'raw')

    # settings
    city_list = ['AZ_Tucson', 'KS_Colwich_Maize', 'NC_Clyde', 'NC_Wilmington']
    tile_range = [12, 15, 8, 12]

    plt.figure(figsize=(10, 6))
    for city_id in range(4):
        dist = []
        for tile_id in range(3, tile_range[city_id]):
            csv_file_name = os.path.join(raw_dir, 'USA_{}_{}.csv'.format(city_list[city_id], tile_id))
            raw_rgb = ersa_utils.load_file(os.path.join(raw_dir, 'USA_{}_{}.tif'.format(city_list[city_id], tile_id)))

            # get tower truth
            gt_list = []
            csv_file_name = os.path.join(raw_dir, 'USA_{}_{}.csv'.format(city_list[city_id], tile_id))
            for label, bbox in read_polygon_csv_data(csv_file_name):
                y, x = get_center_point(*bbox)
                gt_list.append([y, x])

            # visualize results
            for start, stop, online_points in read_line_csv_data(csv_file_name, gt_list):
                centers = add_point_if_not_nearby(start, stop, [gt_list[a] for a in online_points])

                for i in range(len(centers) - 1):
                    dist.append(np.linalg.norm(np.array(centers[i]) - np.array(centers[i+1])))
        plt.subplot(2, 2, city_id+1)
        plt.hist(dist, bins=100)
        plt.xlabel('Distance')
        plt.ylabel('Cnts')
        plt.title(city_list[city_id])
    plt.tight_layout()
    plt.show()
