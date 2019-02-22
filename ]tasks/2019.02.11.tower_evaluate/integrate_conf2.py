import os
import scipy.spatial
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import utils
import ersa_utils
from evaluate_utils import local_maxima_suppression
from integrate_confidence import get_points_between


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

    def get_total_connection(connect_dict, pair):
        return len(connect_dict[pair[0]]) + len(connect_dict[pair[1]])

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


def visualize_with_connected_pairs(raw_rgb, center_list, connected_pairs, add_fig=False):
    if not add_fig:
        plt.figure(figsize=(12, 8))

    plt.imshow(raw_rgb)
    for pair in connected_pairs:
            plt.plot([center_list[pair[0]][1], center_list[pair[1]][1]],
                     [center_list[pair[0]][0], center_list[pair[1]][0]], 'r', linewidth=1)

    if not add_fig:
        plt.tight_layout()
        plt.show()


def prune_pairs(cps, centers):
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
                        # print('removed connection {}'.format(cp2remove))
                        cps.remove(cp2remove)
                        removed_cps.append(cp2remove)
                        flag = True
                        break
    return cps, removed_cps


if __name__ == '__main__':
    # directories
    img_dir, task_dir = utils.get_task_img_folder()
    data_dir = r'/home/lab/Documents/bohao/data/transmission_line'
    raw_dir = os.path.join(data_dir, 'raw')
    conf_dir = r'/media/ei-edl01/user/bh163/tasks/2018.11.16.transmission_line/' \
               r'confmap_uab_UnetCrop_lines_pw30_0_PS(572, 572)_BS5_EP100_LR0.0001_DS60_DR0.1_SFN32'
    lines_dir = r'/media/ei-edl01/data/uab_datasets/lines/data/Original_Tiles'

    city_list = ['AZ_Tucson', 'KS_Colwich_Maize', 'NC_Clyde', 'NC_Wilmington']
    model_name = 'faster_rcnn'

    # settings
    radius = 1500
    width = 7
    th = 5

    for city_id in [0]:
        for tile_id in [1]:
            # load data
            pred_file_name = os.path.join(task_dir, model_name, 'USA_{}_{}.txt'.format(city_list[city_id], tile_id))
            preds = ersa_utils.load_file(pred_file_name)
            csv_file_name = os.path.join(raw_dir, 'USA_{}_{}.csv'.format(city_list[city_id], tile_id))
            raw_rgb = ersa_utils.load_file(os.path.join(raw_dir, 'USA_{}_{}.tif'.format(city_list[city_id], tile_id)))
            conf_img = ersa_utils.load_file(os.path.join(conf_dir, '{}{}.png'.format(city_list[city_id].split('_')[1],
                                                                                     tile_id)))
            line_gt = ersa_utils.load_file(os.path.join(lines_dir, '{}{}_GT.png'.format(city_list[city_id].split('_')[1],
                                                                                        tile_id)))

            # get tower locations
            center_list, conf_list, _ = local_maxima_suppression(preds, 100)

            # get line confidences
            pairs, dists, confs = get_edge_info(center_list, conf_img, radius=radius, width=width,
                                                tile_min=(0, 0), tile_max=raw_rgb.shape)

            # connect lines
            connected_pairs = connect_lines(pairs, confs, th)
            prune_pairs(connected_pairs, center_list)

            # visualize results
            '''plt.figure(figsize=(14, 7))
            ax1 = plt.subplot(121)
            visualize_with_connected_pairs(raw_rgb, center_list, connected_pairs, add_fig=True)
            ax2 = plt.subplot(122, sharex=ax1, sharey=ax1)
            #kernel = np.ones((11, 11), np.uint8)
            #dilation = cv2.dilate(line_gt, kernel, iterations=1)
            plt.imshow(line_gt)
            plt.tight_layout()
            # plt.savefig(os.path.join(img_dir, '{}{}_line_base.png'.format(city_list[city_id], tile_id)))
            plt.show()'''
