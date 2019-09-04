import sys
sys.path.append(r'/home/lab/Documents/bohao/code/third_party/models/research')
sys.path.append(r'/home/lab/Documents/bohao/code/third_party/models/research/object_detection')

import os
import cv2
import copy
import scipy.spatial
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.draw import polygon
import ersa_utils
import util_functions
from nn import nn_utils
from object_detection.utils import ops as utils_ops
from evaluate_utils import get_center_point, local_maxima_suppression


city_list = ['AZ_Tucson', 'KS_Colwich_Maize', 'NC_Clyde', 'NC_Wilmington']
city_list2 = ['Dunedin', 'Gisborne', 'Palmerston North', 'Rotorua', 'Tauranga']


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
            if (x != []) and (y != []):
                yield label, get_bounding_box(y, x)


def load_data(dirs, model_name, city_id, tile_id, merge_range=100):
    conf_dict = {0: 2, 1: 1, 2: 0, 3: 3}
    pred_file_name = os.path.join(dirs['task'], model_name, 'USA_{}_{}.txt'.format(city_list[city_id], tile_id))
    preds = ersa_utils.load_file(pred_file_name)
    raw_rgb = ersa_utils.load_file(os.path.join(dirs['raw'], 'USA_{}_{}.tif'.format(city_list[city_id], tile_id)))
    conf_img = ersa_utils.load_file(os.path.join(dirs['conf'].format(conf_dict[city_id]),
                                                 '{}{}.png'.format(city_list[city_id].split('_')[1], tile_id)))
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
            try:
                conf += conf_map[p[0]][p[1]]
            except IndexError:
                # don't worry, this is because after merging the point could be outside
                # print(p)
                pass
        conf /= len(points)
        line_conf.append(conf)

    assert len(linked_pairs) == len(dist_list) == len(line_conf)

    return linked_pairs, dist_list, line_conf


def order_pair(p1, p2):
    if p1 < p2:
        return p1, p2
    else:
        return p2, p1


def connect_lines(linked_pairs, line_conf, th, cut_n=2):
    def compute_weight(n):
        return th * (n + 1)

    def get_total_connection(cd, pr):
        return len(cd[pr[0]]) + len(cd[pr[1]])

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


def find_close_points(p1, p2, points, th=10):
    def dist2line(p1, p2, p3):
        p1, p2, p3 = np.array(p1), np.array(p2), np.array(p3)
        return np.linalg.norm(np.cross(p2 - p1, p3 - p1))/np.linalg.norm(p2 - p1)

    def is_between(p1, p2, p3):
        # don't change the order of p3
        p1, p2, p3 = np.array(p1), np.array(p2), np.array(p3)
        if 1 - scipy.spatial.distance.cosine((p1 - p3), (p2 - p3)) < 0:
            return True
        else:
            return False

    point_ids = []
    for cnt, p3 in enumerate(points):
        if dist2line(p1, p2, p3) < th and is_between(p1, p2, p3):
            point_ids.append(cnt)

    return point_ids


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


def break_lines(cps, centers):
    unconverge_flag = True
    while unconverge_flag:
        unconverge_flag = False
        for cp in cps:
            online_points = find_close_points(centers[cp[0]], centers[cp[1]], centers)
            if len(online_points) > 0:
                unconverge_flag = True
                cps.remove(cp)
                cps.append((cp[0], online_points[0]))
                for i in range(len(online_points) - 1):
                    cps.append((online_points[i], online_points[i+1]))
                cps.append((online_points[-1], cp[1]))


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


def update_connected_pairs(connected_pairs, tower_pred, connected_towers):
    map_idx = np.arange(len(tower_pred))
    for a in range(len(tower_pred)):
        if a not in connected_towers:
            map_idx[a:] -= 1
    new_cp = []
    for cp in connected_pairs:
        new_cp.append((map_idx[cp[0]], map_idx[cp[1]]))
    return new_cp


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


def visualize_results(img_dir, city_id, tile_id, raw_rgb, line_gt, tower_pred, tower_gt, connected_pairs, connected_towers,
                      unconnected_towers, save_fig=False, post_str='', close_file=False):
    plt.figure(figsize=(8.5, 8))
    img_with_line = util_functions.add_mask(raw_rgb, line_gt, [0, 255, 0], 1)
    visualize_with_connected_pairs(img_with_line, tower_pred, connected_pairs, add_fig=True)
    add_points(tower_gt, 'b', marker='s', size=80, alpha=1, edgecolor='k')
    if connected_towers is not None and unconnected_towers is not None:
        add_points([tower_pred[a] for a in connected_towers], 'r', marker='o', alpha=1, edgecolor='k')
        try:
            add_points([tower_pred[a] for a in unconnected_towers], 'r', marker='o', alpha=1, edgecolor='k')
        except IndexError:
            print('No more unconnected towers')
    else:
        add_points(tower_pred, 'r', marker='o', alpha=1, edgecolor='k')
    plt.axis('off')
    plt.tight_layout()
    if save_fig:
        if 'NZ' in post_str:
            plt.title('{}_{}'.format(city_list2[city_id], tile_id))
            plt.savefig(os.path.join(img_dir, '{}_{}_post_result{}.png'.format(city_list2[city_id], tile_id,
                                                                               post_str)))
        else:
            plt.title('{}_{}'.format(city_list[city_id], tile_id))
            plt.savefig(os.path.join(img_dir, '{}_{}_post_result{}.png'.format(city_list[city_id], tile_id,
                                                                                     post_str)))
    if close_file:
        plt.close()
    else:
        plt.show()


def probability_hough(raw_rgb, connected_pairs, center_list):
    line_graph = np.zeros(raw_rgb.shape[:2], np.uint8)

    for cp in connected_pairs:
        cv2.line(line_graph,
                 tuple([int(a) for a in center_list[cp[0]]]),
                 tuple([int(a) for a in center_list[cp[1]]]),
                 (255,), thickness=5)

    edges = cv2.Canny(line_graph, 50, 150, apertureSize=3)
    minLineLength = 100
    maxLineGap = 10
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength, maxLineGap)

    plt.imshow(raw_rgb)
    print(len(lines))
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            plt.plot([y1, y2], [x1, x2], 'r')
    plt.show()


def linked_length(tower_pred, connected_pairs, dirs, city_id, tile_id, attention_th=2.5):
    line_length_list = []
    attention_pair = []
    for cp in connected_pairs:
        line_length = np.linalg.norm(tower_pred[cp[0]] - tower_pred[cp[1]])
        line_length_list.append(line_length)

        '''if line_length > attention_th:
            attention_pair.append(cp)

            plt.figure(figsize=(8.5, 8))
            plt.imshow(raw_rgb)
            add_points(tower_pred, 'b', marker='s', size=80, alpha=1, edgecolor='k')
            plt.plot([tower_pred[cp[0]][1], tower_pred[cp[1]][1]], [tower_pred[cp[0]][0], tower_pred[cp[1]][0]],
                     'r', linewidth=5)
            plt.tight_layout()
            plt.show()'''

    line_length_list = np.array(line_length_list)
    std = np.std(line_length_list)
    mu = np.mean(line_length_list)
    mal_dist = (line_length_list - mu) / std

    for mal_dist, cp in zip(mal_dist, connected_pairs):
        if mal_dist > attention_th:
            attention_pair.append(cp)

    '''plt.hist((np.array(line_length_list)-mu)/std, bins=100)
    plt.savefig(os.path.join(dirs['image'], '{}_{}_lenhist.png'.format(city_id, tile_id)))
    plt.close()'''

    return mal_dist.tolist(), attention_pair


def get_samples_between(raw_rgb, p1, p2, step, size):
    p1 = (int(p1[0]), int(p1[1]))
    p2 = (int(p2[0]), int(p2[1]))
    y_steps = list(np.linspace(p1[0], p2[0], step, endpoint=False, dtype=int))
    x_steps = list(np.linspace(p1[1], p2[1], step, endpoint=False, dtype=int))

    for y, x in zip(y_steps[1:], x_steps[1:]):
        top_left = (y - size[0]//2, x - size[1]//2)
        sample_patch = raw_rgb[top_left[0]:top_left[0]+size[0], top_left[1]:top_left[1]+size[1], :]
        yield sample_patch, top_left


def towers_online(tower_pred, connected_towers, unconnected_towers, connected_pairs):
    tower_pred = [(a[0], a[1]) for a in tower_pred]
    cts = [(tower_pred[a][0], tower_pred[a][1]) for a in connected_towers]

    connected_towers_temp = copy.deepcopy(connected_towers)
    unconnected_towers_temp = copy.deepcopy(unconnected_towers)
    connected_pairs_temp = copy.deepcopy(connected_pairs)

    # search for nearby towers of unconnected towers
    tree =scipy.spatial.KDTree(cts)
    for ut in unconnected_towers:
        _, near_id = tree.query((tower_pred[ut][0], tower_pred[ut][1]))
        origin_id = tower_pred.index(cts[near_id])
        history_id = -1

        refer_towers = [ut, origin_id]
        for i in range(3):
            for cp in connected_pairs:
                if origin_id in cp and history_id not in cp:
                    if cp[0] == origin_id:
                        refer_towers.append(cp[1])
                        history_id = origin_id
                        origin_id = cp[1]
                    else:
                        refer_towers.append(cp[0])
                        history_id = origin_id
                        origin_id = cp[0]
                    break

        add_flag = False
        if len(refer_towers) >= 3:
            # have enough connections
            for i in range(len(refer_towers) - 2):
                vec_1 = np.array(tower_pred[refer_towers[i + 1]]) - np.array(tower_pred[refer_towers[i]])
                vec_2 = np.array(tower_pred[refer_towers[i + 2]]) - np.array(tower_pred[refer_towers[i + 1]])

                angle = np.abs(np.math.atan2(np.linalg.det([vec_1, vec_2]), np.dot(vec_1, vec_2))) / np.pi * 180
                if angle > 90:
                    angle = 180 - angle

                if angle > 5:
                    break

                if i == len(refer_towers) - 3:
                    add_flag = True

        if add_flag:
            connected_towers_temp.append(ut)
            unconnected_towers_temp.remove(ut)
            connected_pairs_temp.append(order_pair(ut, origin_id))

            '''plt.figure(figsize=(8.5, 8))
            plt.imshow(raw_rgb)
            add_points([tower_pred[a] for a in connected_towers], 'b', marker='o', alpha=1, edgecolor='k')
            add_points([tower_pred[a] for a in refer_towers], 'yellow', marker='s', size=80, alpha=1, edgecolor='k')
            add_points([tower_pred[ut]], 'r', marker='o', alpha=1, edgecolor='k')
            plt.tight_layout()
            plt.show()'''

    return connected_towers_temp, unconnected_towers_temp, connected_pairs_temp


def run_inference_for_single_image(image, graph):
    with graph.as_default():
        with tf.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                        tensor_name)
            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, image.shape[0], image.shape[1])
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(
                    detection_masks_reframed, 0)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = sess.run(tensor_dict,
                                   feed_dict={image_tensor: np.expand_dims(image, 0)})

            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict[
                'detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict


def load_model(model_name, model_id, GPU='0'):
    from utils import label_map_util
    path_to_frozen_graph = r'/hdd6/Models/transmission_line/' \
                           r'export_model/{}/{}/frozen_inference_graph.pb'.format(model_name, model_id)
    path_to_labels = r'/home/lab/Documents/bohao/data/transmission_line/data/label_map_t.pbtxt'
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU)
    nn_utils.tf_warn_level(3)

    # load frozen tf model into memory
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(path_to_frozen_graph, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    # load label map
    category_index = label_map_util.create_category_index_from_labelmap(path_to_labels, use_display_name=True)
    return detection_graph, category_index
