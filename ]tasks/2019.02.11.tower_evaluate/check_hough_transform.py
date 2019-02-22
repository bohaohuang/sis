import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import utils
import ersa_utils
from evaluate_utils import local_maxima_suppression
from evaluate_tower_performance import get_center_point, read_polygon_csv_data
from integrate_conf2 import prune_pairs, get_edge_info, connect_lines


def get_points_between(raw_rgb, point_1, point_2, width=7, tile_min=(0, 0), tile_max=(5000, 5000)):
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

    patch = np.zeros_like(raw_rgb)
    for p in points:
        patch[p[0], p[1]] = raw_rgb[p[0], p[1]]

    y = [top_left[0], top_right[0], bot_left[0], bot_right[0]]
    x = [top_left[1], top_right[1], bot_left[1], bot_right[1]]
    y_min, y_max, x_min, x_max = min(y), max(y), min(x), max(x)

    return patch[y_min:y_max, x_min:x_max]


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

    city_id, tile_id = 0, 1
    # load data
    pred_file_name = os.path.join(task_dir, model_name, 'USA_{}_{}.txt'.format(city_list[city_id], tile_id))
    preds = ersa_utils.load_file(pred_file_name)
    csv_file_name = os.path.join(raw_dir, 'USA_{}_{}.csv'.format(city_list[city_id], tile_id))
    raw_rgb = ersa_utils.load_file(os.path.join(raw_dir, 'USA_{}_{}.tif'.format(city_list[city_id], tile_id)))
    conf_img = ersa_utils.load_file(os.path.join(conf_dir, '{}{}.png'.format(city_list[city_id].split('_')[1],
                                                                             tile_id)))
    line_gt = ersa_utils.load_file(os.path.join(lines_dir, '{}{}_GT.png'.format(city_list[city_id].split('_')[1],
                                                                                tile_id)))

    # get tower preds
    center_list, conf_list, _ = local_maxima_suppression(preds, 100)

    # get tower truth
    gt_list = []
    csv_file_name = os.path.join(raw_dir, 'USA_{}_{}.csv'.format(city_list[city_id], tile_id))
    for label, bbox in read_polygon_csv_data(csv_file_name):
        y, x = get_center_point(*bbox)
        gt_list.append([y, x])

    # get line confidences
    pairs, dists, confs = get_edge_info(center_list, conf_img, radius=radius, width=width,
                                        tile_min=(0, 0), tile_max=raw_rgb.shape)

    # connect lines
    connected_pairs = connect_lines(pairs, confs, th)
    connected_pairs, unconnected_pairs = prune_pairs(connected_pairs, center_list)

    gray = cv2.cvtColor(raw_rgb, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, 500, 20)
    print(len(lines), len(lines[0]), lines[0])

    plt.imshow(raw_rgb)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            plt.plot([x1, x2], [y1, y2], 'r')
    plt.show()

    '''for cp in connected_pairs:
        print(center_list[cp[0]], center_list[cp[1]])
        patch = \
            get_points_between(raw_rgb, center_list[cp[0]], center_list[cp[1]], width=7, tile_min=(0, 0),
                               tile_max=raw_rgb.shape)

        # extract line
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, 500, 50)

        plt.imshow(patch)
        if lines is not None:
            for x1,y1,x2,y2 in lines[0]:
                plt.plot([x1, x2], [y1, y2], 'r')
            plt.show()
        # break'''
