import os
import numpy as np
import scipy.spatial
import matplotlib.pyplot as plt
from tqdm import tqdm
import utils
import ersa_utils
from evaluate_utils import local_maxima_suppression


def get_link_matrix(linked_items, centers, radius):
    n = len(centers)
    link_matrix = np.zeros((n, n))
    for pair in linked_items:
        if np.linalg.norm(centers[pair[0]] - centers[pair[1]]) < radius:
            link_matrix[pair[0], pair[1]] = 1
    return link_matrix


def count_connected_items(linked_items):
    count_list = []
    for cnt, i in enumerate(linked_items):
        count_list.append(len([a for a in i if a != cnt]))
    return count_list


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


def get_connect_matrix(centers, conf_map, radius=1500, width=7, tile_min=(0, 0),
                       tile_max=(5000, 5000)):
    # link towers
    kdt = scipy.spatial.KDTree(np.array(centers))
    linked_items = kdt.query_pairs(radius)

    for i in linked_items:
        assert np.linalg.norm(centers[i[0]] - centers[i[1]]) < radius

    # count number of towers connected
    link_matrix = get_link_matrix(linked_items, centers, radius)
    line_matrix = np.zeros_like(link_matrix)
    line_conf = []
    line_pois = []
    n, _ = link_matrix.shape

    for i in tqdm(range(n)):
        for j in range(i+1, n):
            if link_matrix[i][j] == 1:
                points = get_points_between(centers[i], centers[j], width=width, tile_min=tile_min,
                                            tile_max=tile_max)
                conf = 0
                for p in points:
                    conf += conf_map[p[0]][p[1]]
                conf /= len(points)
                line_matrix[i][j] = conf
                line_conf.append(conf)
                line_pois.append([i, j])

    # verify link matrix distance
    for i in range(n):
        for j in range(i + 1, n):
            if line_matrix[i][j] > 0:
                assert np.linalg.norm(centers[i] - centers[j]) < radius

    return line_conf, line_pois, line_matrix


def prune_graph(conf_img, center_points, line_conf, line_pois, tile_max, th=1.5e-3):
    def iou(a, b):
        return np.sum(a * b) / (np.sum(a) + np.sum(b) - np.sum(a * b))

    power_line = np.zeros_like(conf_img, dtype=np.uint8)
    line_pois_prune = []

    sort_idx = np.argsort(line_conf)[::-1]
    old_iou = -1
    for i in tqdm(sort_idx):
        power_line_temp = np.copy(power_line)
        ps = get_points_between(center_points[line_pois[i][0]], center_points[line_pois[i][1]],
                                tile_max=tile_max)
        for p in ps:
            power_line_temp[p[0]][p[1]] = 1
        new_iou = iou(conf_img, power_line_temp)
        if new_iou > old_iou + th:
            print('Added point {} to {}, IoU now is {:.2f}'.format(line_pois[i][0], line_pois[i][1],
                                                                   new_iou))
            old_iou = new_iou
            power_line = np.copy(power_line_temp)
            del power_line_temp
            line_pois_prune.append(line_pois[i])

    return line_pois_prune, power_line


def connect_points(line_matrix, th=20):
    def connect_sum(cm, i, j):
        return np.sum(cm[i, :]) + np.sum(cm[:, j]) - cm[i, j]

    def connect_sum_max(cm, i, j):
        n = cm.shape[0]
        max_connect = []
        for v in range(n):
            c = connect_sum(cm, v, j)
            if c > 0:
                max_connect.append(c)
        for h in range(n):
            c = connect_sum(cm, i, h)
            if c > 0:
                max_connect.append(c)
        return np.mean(max_connect)

    def compute_weight(n_connect):
        return (np.exp(n_connect)) * th

    connect_matrix = np.zeros_like(line_matrix, dtype=np.uint8)

    # sort array
    sort_idx_all = np.unravel_index(np.argsort(line_matrix, axis=None), line_matrix.shape)
    sort_idx_all = [sort_idx_all[0].tolist(), sort_idx_all[1].tolist()]

    # find first connection
    max_ind = (sort_idx_all[0][-1], sort_idx_all[1][-1])
    connect_matrix[max_ind] = 1
    orientation = 0

    current_ind = max_ind
    while True:
        if orientation == 0:
            # search rows
            sort_idx = np.argsort(line_matrix[current_ind[0], :])[::-1]
            current_connects = connect_sum(connect_matrix, current_ind[0], current_ind[1])
            if current_connects < 2:
                # not enough connection yet
                for i in sort_idx[1:]:
                    if connect_matrix[current_ind[0], i] != 1 and line_matrix[current_ind[0], i] > th:
                        connect_matrix[current_ind[0], i] = 1
                        current_ind = (current_ind[0], i)
                        break
            else:
                # already have enough connection
                restart_flag = True
                for i in sort_idx[1:]:
                    if connect_matrix[current_ind[0], i] != 1:
                        if line_matrix[current_ind[0], i] > compute_weight(current_connects):
                            connect_matrix[current_ind[0], i] = 1
                            current_ind = (current_ind[0], i)
                            restart_flag = False
                            break
                if restart_flag:
                    return_flag = True
                    # try find new start point
                    for i, j in zip(sort_idx_all[0][::-1], sort_idx_all[1][::-1]):
                        if connect_matrix[i, j] == 0 and line_matrix[i, j] > compute_weight(connect_sum(connect_matrix, i, j)):
                                # and connect_sum_max(connect_matrix, i, j) < 2:
                            orientation = 1
                            connect_matrix[i, j] = 1
                            current_ind = (i, j)
                            return_flag = False
                            break
                    if return_flag:
                        return connect_matrix
            orientation = 1 - orientation
        elif orientation == 1:
            # search cols
            sort_idx = np.argsort(line_matrix[:, current_ind[1]])[::-1]
            current_connects = connect_sum(connect_matrix, current_ind[0], current_ind[1])
            if current_connects < 2:
                # not enough connection yet
                for i in sort_idx[1:]:
                    if connect_matrix[i, current_ind[1]] != 1 and line_matrix[current_ind[0], i] > th:
                        connect_matrix[i, current_ind[1]] = 1
                        current_ind = (i, current_ind[1])
                        break
            else:
                # already have enough connection
                restart_flag = True
                for i in sort_idx[1:]:
                    if connect_matrix[i, current_ind[1]] != 1:
                        if line_matrix[i, current_ind[1]] > compute_weight(current_connects):
                            connect_matrix[i, current_ind[1]] = 1
                            current_ind = (i, current_ind[1])
                            restart_flag = False
                            break
                if restart_flag:
                    return_flag = True
                    # try find new start point
                    for i, j in zip(sort_idx_all[0][::-1], sort_idx_all[1][::-1]):
                        if connect_matrix[i, j] == 0 and line_matrix[i, j] > compute_weight(connect_sum(connect_matrix, i, j)):
                                #and connect_sum_max(connect_matrix, i, j) < 2:
                            orientation = 1
                            connect_matrix[i, j] = 1
                            current_ind = (i, j)
                            return_flag = False
                            break
                    if return_flag:
                        return connect_matrix
            orientation = 1 - orientation


def visuliaze_with_connect_matrix(raw_rgb, center_list, connect_matrix, add_fig=False):
    n, _ = connect_matrix.shape

    if not add_fig:
        plt.figure(figsize=(12, 8))

    plt.imshow(raw_rgb)
    for i in range(n):
        for j in range(i+1, n):
            if connect_matrix[i][j] == 1:
                plt.plot([center_list[i][1], center_list[j][1]],
                         [center_list[i][0], center_list[j][0]], 'r', linewidth=1)

    if not add_fig:
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    img_dir, task_dir = utils.get_task_img_folder()
    data_dir = r'/home/lab/Documents/bohao/data/transmission_line'
    raw_dir = os.path.join(data_dir, 'raw')
    conf_dir = r'/media/ei-edl01/user/bh163/tasks/2018.11.16.transmission_line/' \
               r'confmap_uab_UnetCrop_lines_pw30_0_PS(572, 572)_BS5_EP100_LR0.0001_DS60_DR0.1_SFN32'
    lines_dir = r'/media/ei-edl01/data/uab_datasets/lines/data/Original_Tiles'

    city_list = ['AZ_Tucson', 'KS_Colwich_Maize', 'NC_Clyde', 'NC_Wilmington']

    for city_id in [3]:
        for tile_id in [1, 2, 3]:
            model_name = 'faster_rcnn'

            # load data
            pred_file_name = os.path.join(task_dir, model_name, 'USA_{}_{}.txt'.format(city_list[city_id], tile_id))
            preds = ersa_utils.load_file(pred_file_name)
            csv_file_name = os.path.join(raw_dir, 'USA_{}_{}.csv'.format(city_list[city_id], tile_id))
            raw_rgb = ersa_utils.load_file(os.path.join(raw_dir, 'USA_{}_{}.tif'.format(city_list[city_id], tile_id)))
            conf_img = ersa_utils.load_file(os.path.join(conf_dir, '{}{}.png'.format(city_list[city_id].split('_')[1],
                                                                                     tile_id)))
            line_gt = ersa_utils.load_file(os.path.join(lines_dir, '{}{}_GT.png'.format(city_list[city_id].split('_')[1],
                                                                                        tile_id)))

            center_list, conf_list, _ = local_maxima_suppression(preds, 100)
            bbox = []
            for line in center_list:
                bbox.append([float(line[1]), float(line[0]), float(line[1])+20, float(line[0])+20])

            line_conf, line_pois, line_matrix = get_connect_matrix(center_list, conf_img, radius=1500, width=7,
                                                                   tile_min=(0, 0), tile_max=raw_rgb.shape)
            connect_matrix = connect_points(line_matrix)

            # check connections
            def connect_sum(cm, i, j):
                return np.sum(cm[i, :]) + np.sum(cm[:, j]) - cm[i, j]
            n = connect_matrix.shape[0]
            more_cnt = 0
            for i in range(n):
                for j in range(i+1, n):
                    if connect_sum(connect_matrix, i, j) > 2:
                        more_cnt += 1
            print(more_cnt)

            # visualize results
            plt.figure(figsize=(14, 7))
            ax1 = plt.subplot(121)
            visuliaze_with_connect_matrix(raw_rgb, center_list, connect_matrix, add_fig=True)
            ax2 = plt.subplot(122, sharex=ax1, sharey=ax1)
            plt.imshow(line_gt)
            plt.tight_layout()
            #plt.savefig(os.path.join(img_dir, '{}{}_line_base.png'.format(city_list[city_id], tile_id)))
            plt.show()
