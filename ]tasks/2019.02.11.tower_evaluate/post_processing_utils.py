import os
import scipy.spatial
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.draw import polygon
import ersa_utils
import util_functions
from evaluate_utils import get_center_point, local_maxima_suppression


city_list = ['AZ_Tucson', 'KS_Colwich_Maize', 'NC_Clyde', 'NC_Wilmington']


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


def load_data(dirs, model_name, city_id, tile_id, merge_range=100):
    pred_file_name = os.path.join(dirs['task'], model_name, 'USA_{}_{}.txt'.format(city_list[city_id], tile_id))
    preds = ersa_utils.load_file(pred_file_name)
    raw_rgb = ersa_utils.load_file(os.path.join(dirs['raw'], 'USA_{}_{}.tif'.format(city_list[city_id], tile_id)))
    conf_img = ersa_utils.load_file(os.path.join(dirs['conf'], '{}{}.png'.format(city_list[city_id].split('_')[1],
                                                                                 tile_id)))
    line_gt = ersa_utils.load_file(os.path.join(dirs['line'], '{}{}_GT.png'.format(city_list[city_id].split('_')[1],
                                                                                   tile_id)))
    tower_gt = get_tower_truth_pred(dirs, city_id, tile_id)
    tower_pred, tower_conf, _ = local_maxima_suppression(preds, th=merge_range)
    return preds, raw_rgb, conf_img, line_gt, tower_gt, tower_pred, tower_conf


def get_tower_truth_pred(dirs, city_id, tile_id):
    gt_list = []
    csv_file_name = os.path.join(dirs['raw'], 'USA_{}_{}.csv'.format(city_list[city_id], tile_id))
    for label, bbox in read_polygon_csv_data(csv_file_name):
        y, x = get_center_point(*bbox)
        gt_list.append([y, x])
    return gt_list


def get_points_between(point_1, point_2, width=7, tile_min=(0, 0), tile_max=(5000, 5000)):
    def p_between(p1, p2):
        p1 = np.array(p1)
        p2 = np.array(p2)
        d = np.linalg.norm(p1 - p2)
        n_sample = np.round(d)
        return list(zip(np.linspace(p1[0], p2[0], n_sample + 1, dtype=np.int),
                        np.linspace(p1[1], p2[1], n_sample + 1, dtype=np.int)))

    def move_in_tile(p, min_val, max_val):
        for i in range(2):
            p[i] = np.max([min_val[i], p[i]])
            p[i] = np.min([max_val[i], p[i]])
        return p

    point_1 = point_1.astype(np.int)
    point_2 = point_2.astype(np.int)

    # get corner points
    p_vec = np.array([-point_2[1]+point_1[1], point_2[0]-point_1[0]])
    p_vec = p_vec / np.linalg.norm(p_vec)
    top_left = move_in_tile((point_1 + p_vec * width).astype(np.int), tile_min, tile_max)
    top_right = move_in_tile((point_2 + p_vec * width).astype(np.int), tile_min, tile_max)
    bot_left = move_in_tile((point_1 - p_vec * width).astype(np.int), tile_min, tile_max)
    bot_right = move_in_tile((point_2 - p_vec * width).astype(np.int), tile_min, tile_max)

    l_1 = p_between(top_left, bot_left)
    l_2 = p_between(top_right, bot_right)
    points = []
    for p1, p2 in zip(l_1, l_2):
        points.extend(p_between(p1, p2))

    return points


def get_edge_info(centers, conf_map, radius=1500, width=7, tile_min=(0, 0), tile_max=(5000, 5000)):
    # link towers
    kdt = scipy.spatial.KDTree(np.array(centers))
    linked_pairs = list(kdt.query_pairs(radius))

    # sort by distance
    dist_list = []
    for pair in linked_pairs:
        dist_list.append(np.linalg.norm(centers[pair[0]] - centers[pair[1]]))
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


def connect_lines(linked_pairs, line_conf, th, cut_n=2):
    def compute_weight(n):
        return th * (n + 1)

    def get_total_connection(cd, pr):
        return len(cd[pr[0]]) + len(cd[pr[1]])

    def order_pair(p1, p2):
        if p1 < p2:
            return p1, p2
        else:
            return p2, p1

    def get_conf(p1, p2):
        return line_conf[linked_pairs.index(order_pair(p1, p2))]

    # get connection dict
    connect_dict = {}
    connected_pair = []
    for pair in linked_pairs:
        if pair[0] not in connect_dict:
            connect_dict[pair[0]] = []
        if pair[1] not in connect_dict:
            connect_dict[pair[1]] = []

    # determine if connect two towers or not
    for cnt, pair in enumerate(linked_pairs):
        if line_conf[cnt] > compute_weight(get_total_connection(connect_dict, pair)):
            connected_pair.append(pair)
            connect_dict[pair[0]].append(pair[1])
            connect_dict[pair[1]].append(pair[0])
        # cut connections if necessary
        for p in pair:
            if len(connect_dict[p]) > cut_n:
                for temp_pair in connect_dict[p]:
                    if get_conf(p, temp_pair) < compute_weight(get_total_connection(connect_dict, order_pair(p, temp_pair))):
                        # remove connection
                        connect_dict[p].remove(temp_pair)
                        try:
                            connected_pair.remove(order_pair(p, temp_pair))
                        except ValueError:
                            pass

    return connected_pair


def prune_lines(cps, centers):
    def search_cp(p):
        return [a for a in cps if p in a]

    def get_angles(cp3, centers):
        point3 = [cp3[0][0], cp3[0][1],
                  cp3[1][0], cp3[1][1],
                  cp3[2][0], cp3[2][1]]
        point3 = list(set(point3))

        for p in point3:
            other_p = [a for a in point3 if a != p]
            angle = arc_angle(np.array(centers[other_p[0]]) - np.array(centers[p]),
                              np.array(centers[other_p[1]]) - np.array(centers[p]))
            if angle > 0.9:
                return tuple(np.sort(other_p).tolist())

    def arc_angle(p0, p1):
        return np.abs(np.math.atan2(np.linalg.det([p0, p1]), np.dot(p0, p1))) / np.pi

    removed_cps = []
    flag = True
    while flag:
        flag = False
        for cp in cps:
            # always search cp[0]
            target_cps = [a for a in search_cp(cp[0]) if a != cp]
            for target_cp in target_cps:
                if target_cp[0] == cp[0]:
                    p = target_cp[1]
                else:
                    p = target_cp[0]
                if (p, cp[1]) in cps:
                    cp2remove = get_angles([cp, target_cp, (p, cp[1])], centers)
                    if cp2remove is not None:
                        cps.remove(cp2remove)
                        removed_cps.append(cp2remove)
                        flag = True
                        break
    return cps, removed_cps


def prune_towers(connected_pairs, tower_pred):
    kept_towers = []
    for p in connected_pairs:
        kept_towers.append(p[0])
        kept_towers.append(p[1])
    connected_towers = list(set(kept_towers))
    unconnected_towers = [a for a in range(len(tower_pred)) if a not in connected_towers]

    return connected_towers, unconnected_towers


def add_points(center_points, color='r', size=20, marker='o', alpha=0.5, edgecolor='face'):
    center_points = np.array(center_points)
    plt.scatter(center_points[:, 1], center_points[:, 0], c=color, s=size, marker=marker, alpha=alpha,
                edgecolors=edgecolor)


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


def visualize_results(dirs, city_id, tile_id, raw_rgb, line_gt, tower_pred, tower_gt, connected_pairs, connected_towers,
                      unconnected_towers, save_fig=False):
    plt.figure(figsize=(8.5, 8))
    img_with_line = util_functions.add_mask(raw_rgb, line_gt, [0, 255, 0], 1)
    visualize_with_connected_pairs(img_with_line, tower_pred, connected_pairs, add_fig=True)
    add_points(tower_gt, 'b', marker='s', size=80, alpha=1, edgecolor='k')
    add_points([tower_pred[a] for a in connected_towers], 'r', marker='o', alpha=1, edgecolor='k')
    try:
        add_points([tower_pred[a] for a in unconnected_towers], 'yellow', marker='o', alpha=1, edgecolor='k')
    except IndexError:
        print('No more unconnected towers')
    plt.axis('off')
    plt.title('{}_{}'.format(city_list[city_id], tile_id))
    plt.tight_layout()
    if save_fig:
        plt.savefig(os.path.join(dirs['image'], '{}_{}_post_result.png'.format(city_list[city_id], tile_id)))
    plt.show()
